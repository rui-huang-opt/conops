use std::{
    collections::HashMap,
    net::{SocketAddr, SocketAddrV4},
};

use mdns_sd::{Receiver, ServiceDaemon, ServiceEvent, ServiceInfo};

pub const SERVICE_TYPE: &str = "_conops._tcp.local.";

#[derive(Debug)]
pub struct DiscoveredPeer {
    pub node_id: String,
    pub addresses: Vec<SocketAddr>,
}

#[derive(Debug)]
pub enum DiscoveryEvent {
    Found(DiscoveredPeer),
    Removed(String),
}

pub struct Discovery {
    daemon: ServiceDaemon,
    events: Receiver<ServiceEvent>,
    fullname: String,
    node_id: String,
    group_id: String,
    known: HashMap<String, String>,
}

impl Discovery {
    /// Publish this node and discover peers in the same group.
    pub fn new(node_id: &str, port: u16, group_id: &str) -> mdns_sd::Result<Self> {
        let daemon = ServiceDaemon::new()?;

        let hostname = format!("conops-{}-{port}.local.", std::process::id());

        let properties = [("node_id", node_id), ("group", group_id)];

        let service =
            ServiceInfo::new(SERVICE_TYPE, node_id, &hostname, "", port, &properties[..])?
                .enable_addr_auto();

        let fullname = service.get_fullname().to_owned();

        daemon.register(service)?;
        let events = daemon.browse(SERVICE_TYPE)?;

        Ok(Self {
            daemon,
            events,
            fullname,
            node_id: node_id.to_owned(),
            group_id: group_id.to_owned(),
            known: HashMap::new(),
        })
    }

    /// Wait for the next relevant discovery event.
    pub async fn next(&mut self) -> Option<DiscoveryEvent> {
        loop {
            let event = self.events.recv_async().await.ok()?;

            match event {
                ServiceEvent::ServiceResolved(info) => {
                    let Some(peer_id) = info.get_property_val_str("node_id") else {
                        continue;
                    };

                    // Ignore self and other experiment groups.
                    if peer_id == self.node_id
                        || info.get_property_val_str("group") != Some(self.group_id.as_str())
                    {
                        continue;
                    }

                    let mut addresses: Vec<_> = info
                        .get_addresses_v4()
                        .into_iter()
                        .filter(|ip| !ip.is_loopback())
                        .map(|ip| SocketAddr::V4(SocketAddrV4::new(ip, info.get_port())))
                        .collect();

                    addresses.sort();

                    if addresses.is_empty() {
                        continue;
                    }

                    self.known
                        .insert(info.get_fullname().to_owned(), peer_id.to_owned());

                    return Some(DiscoveryEvent::Found(DiscoveredPeer {
                        node_id: peer_id.to_owned(),
                        addresses,
                    }));
                }

                ServiceEvent::ServiceRemoved(_, fullname) => {
                    if let Some(node_id) = self.known.remove(&fullname) {
                        return Some(DiscoveryEvent::Removed(node_id));
                    }
                }

                _ => {}
            }
        }
    }

    /// Stop advertising and browsing.
    pub fn close(self) -> mdns_sd::Result<()> {
        self.daemon.stop_browse(SERVICE_TYPE)?;
        let _ = self.daemon.unregister(&self.fullname)?;
        let _ = self.daemon.shutdown()?;

        Ok(())
    }
}
