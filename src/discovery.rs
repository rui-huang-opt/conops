use std::{
    collections::HashMap,
    io,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use socket2::{Domain, Protocol, Socket, Type};
use tokio::{
    net::UdpSocket,
    time::{Interval, MissedTickBehavior, interval},
};

// Private discovery protocol, independent of ConOps's TCP handshake.
const MULTICAST_IP: Ipv4Addr = Ipv4Addr::new(239, 255, 42, 99);
const MULTICAST_PORT: u16 = 47222;
const SCOUT_INTERVAL: Duration = Duration::from_secs(2);
const MAGIC: &[u8] = b"CNOP";
const VERSION: u8 = 1;
const SCOUT: u8 = 1;
const HELLO: u8 = 2;
const HEADER_LEN: usize = 18;

#[derive(Debug)]
pub struct DiscoveredPeer {
    pub node_id: String,
    pub addresses: Vec<SocketAddr>,
}

struct Message<'a> {
    kind: u8,
    group_id: &'a str,
    node_id: &'a str,
    tcp_port: u16,
    instance_id: u64,
}

// 4-byte magic, version, kind, group length, node length, TCP port, instance ID,
// followed by the UTF-8 group and node identifiers.
fn encode_message(message: &Message<'_>) -> io::Result<Vec<u8>> {
    let group_len = u8::try_from(message.group_id.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "group_id is too long"))?;
    let node_len = u8::try_from(message.node_id.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "node_id is too long"))?;

    let mut packet =
        Vec::with_capacity(HEADER_LEN + usize::from(group_len) + usize::from(node_len));
    packet.extend_from_slice(MAGIC);
    packet.extend_from_slice(&[VERSION, message.kind, group_len, node_len]);
    packet.extend_from_slice(&message.tcp_port.to_be_bytes());
    packet.extend_from_slice(&message.instance_id.to_be_bytes());
    packet.extend_from_slice(message.group_id.as_bytes());
    packet.extend_from_slice(message.node_id.as_bytes());
    Ok(packet)
}

fn decode_message(data: &[u8]) -> Option<Message<'_>> {
    if data.len() < HEADER_LEN || &data[..4] != MAGIC || data[4] != VERSION {
        return None;
    }

    let kind = data[5];
    if kind != SCOUT && kind != HELLO {
        return None;
    }

    let group_len = usize::from(data[6]);
    let node_len = usize::from(data[7]);
    if group_len == 0 || node_len == 0 || data.len() != HEADER_LEN + group_len + node_len {
        return None;
    }

    let tcp_port = u16::from_be_bytes(data[8..10].try_into().ok()?);
    if kind == HELLO && tcp_port == 0 {
        return None;
    }

    let instance_id = u64::from_be_bytes(data[10..18].try_into().ok()?);
    let group_id = std::str::from_utf8(&data[HEADER_LEN..HEADER_LEN + group_len]).ok()?;
    let node_id = std::str::from_utf8(&data[HEADER_LEN + group_len..]).ok()?;

    Some(Message {
        kind,
        group_id,
        node_id,
        tcp_port,
        instance_id,
    })
}

pub struct Discovery {
    // All nodes on the same host listen to this multicast port.
    multicast: UdpSocket,
    // A different ephemeral port for each node: receives unicast HELLO replies.
    unicast: UdpSocket,
    ticker: Interval,
    scout_packet: Vec<u8>,
    hello_packet: Vec<u8>,
    node_id: String,
    group_id: String,
    known: HashMap<String, (u64, SocketAddr)>,
}

impl Discovery {
    /// Publish this node's TCP address via HELLO and discover peers by SCOUT.
    /// Must be called inside a Tokio runtime (UdpSocket::from_std requirement).
    pub fn new(node_id: &str, port: u16, group_id: &str) -> io::Result<Self> {
        if node_id.is_empty() || group_id.is_empty() || port == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "node_id, group_id and TCP port must be nonzero/nonempty",
            ));
        }

        // Reuse the multicast port so multiple ConOps processes can coexist.
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        socket.bind(&SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, MULTICAST_PORT).into())?;
        socket.join_multicast_v4(&MULTICAST_IP, &Ipv4Addr::UNSPECIFIED)?;
        socket.set_nonblocking(true)?;
        let multicast = UdpSocket::from_std(socket.into())?;

        // Reply to the unique source port of each SCOUT, not the shared port.
        let unicast = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
        unicast.set_nonblocking(true)?;
        let unicast = UdpSocket::from_std(unicast)?;
        unicast.set_multicast_loop_v4(true)?; // discover other nodes on this host
        unicast.set_multicast_ttl_v4(1)?; // local subnet only

        // Distinguish a restarted process with the same logical node ID.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let instance_id = nanos ^ (u64::from(std::process::id()) << 32);

        let scout_packet = encode_message(&Message {
            kind: SCOUT,
            group_id,
            node_id,
            tcp_port: 0,
            instance_id,
        })?;
        let hello_packet = encode_message(&Message {
            kind: HELLO,
            group_id,
            node_id,
            tcp_port: port,
            instance_id,
        })?;

        let mut ticker = interval(SCOUT_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        Ok(Self {
            multicast,
            unicast,
            ticker,
            scout_packet,
            hello_packet,
            node_id: node_id.to_owned(),
            group_id: group_id.to_owned(),
            known: HashMap::new(),
        })
    }

    /// Wait for a previously unknown or changed peer address.
    pub async fn next(&mut self) -> Option<DiscoveredPeer> {
        let mut multicast_buf = [0u8; 1024];
        let mut unicast_buf = [0u8; 1024];
        let multicast_addr = SocketAddr::V4(SocketAddrV4::new(MULTICAST_IP, MULTICAST_PORT));

        loop {
            tokio::select! {
                _ = self.ticker.tick() => {
                    if let Err(err) = self.unicast.send_to(&self.scout_packet, multicast_addr).await {
                        log::warn!("Scouting SCOUT send failed: {err}");
                    }
                }

                result = self.multicast.recv_from(&mut multicast_buf) => {
                    let (len, sender) = match result {
                        Ok(packet) => packet,
                        Err(err) => {
                            log::warn!("Scouting multicast receive failed: {err}");
                            return None;
                        }
                    };
                    let Some(message) = decode_message(&multicast_buf[..len]) else {
                        continue;
                    };
                    if message.kind != SCOUT
                        || message.group_id != self.group_id.as_str()
                        || message.node_id == self.node_id.as_str()
                    {
                        continue;
                    }
                    if let Err(err) = self.unicast.send_to(&self.hello_packet, sender).await {
                        log::warn!("Scouting HELLO send failed: {err}");
                    }
                }

                result = self.unicast.recv_from(&mut unicast_buf) => {
                    let (len, sender) = match result {
                        Ok(packet) => packet,
                        Err(err) => {
                            log::warn!("Scouting unicast receive failed: {err}");
                            return None;
                        }
                    };
                    let Some(message) = decode_message(&unicast_buf[..len]) else {
                        continue;
                    };
                    if message.kind != HELLO
                        || message.group_id != self.group_id.as_str()
                        || message.node_id == self.node_id.as_str()
                    {
                        continue;
                    }

                    let endpoint = SocketAddr::new(sender.ip(), message.tcp_port);
                    let identity = (message.instance_id, endpoint);
                    if self.known.get(message.node_id) == Some(&identity) {
                        continue;
                    }
                    self.known.insert(message.node_id.to_owned(), identity);

                    return Some(DiscoveredPeer {
                        node_id: message.node_id.to_owned(),
                        addresses: vec![endpoint],
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[test]
    fn test_packet_codec() {
        let packet = encode_message(&Message {
            kind: HELLO,
            group_id: "demo/group",
            node_id: "node-2",
            tcp_port: 5001,
            instance_id: 42,
        })
        .unwrap();

        let message = decode_message(&packet).unwrap();
        assert_eq!(message.kind, HELLO);
        assert_eq!(message.group_id, "demo/group");
        assert_eq!(message.node_id, "node-2");
        assert_eq!(message.tcp_port, 5001);
        assert_eq!(message.instance_id, 42);
        assert!(decode_message(&packet[..packet.len() - 1]).is_none());
    }

    #[tokio::test]
    #[ignore = "requires local UDP multicast support"]
    async fn test_discovery() {
        let pid = std::process::id();

        let a_id = format!("test-a-{pid}");
        let b_id = format!("test-b-{pid}");
        let namespace = format!("test-namespace-{pid}");

        let mut a = Discovery::new(&a_id, 5000, &namespace).unwrap();
        let mut b = Discovery::new(&b_id, 5001, &namespace).unwrap();

        let (found_b, found_a) = timeout(Duration::from_secs(15), async {
            let mut found_b = None;
            let mut found_a = None;

            while found_b.is_none() || found_a.is_none() {
                tokio::select! {
                    event = a.next() => {
                        match event {
                            Some(peer) if peer.node_id == b_id => {
                                found_b = Some(peer);
                            }
                            Some(_) => {}
                            None => panic!("Discovery A stopped unexpectedly"),
                        }
                    }

                    event = b.next() => {
                        match event {
                            Some(peer) if peer.node_id == a_id => {
                                found_a = Some(peer);
                            }
                            Some(_) => {}
                            None => panic!("Discovery B stopped unexpectedly"),
                        }
                    }
                }
            }

            (found_b.unwrap(), found_a.unwrap())
        })
        .await
        .expect("Bidirectional discovery timed out");

        assert_eq!(found_b.addresses[0].port(), 5001);
        assert_eq!(found_a.addresses[0].port(), 5000);
    }
}
