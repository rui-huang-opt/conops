use std::io;

use bytes::{Buf, BufMut, Bytes, BytesMut};

// The outer TCP frame is still a 4-byte big-endian length prefix.
// The inner frame is: 1-byte type + 8-byte exchange ID + optional payload.
const TYPE_DATA: u8 = 0;
const TYPE_ACK: u8 = 1;
pub const HEADER_SIZE: usize = 9;

#[derive(Debug, PartialEq, Eq)]
pub enum Message {
    Data { exchange_id: u64, payload: Bytes },
    Ack { exchange_id: u64 },
}

/// Encode a DATA frame. The returned Bytes can be cheaply cloned for peers.
pub fn encode(exchange_id: u64, payload: &[u8]) -> Bytes {
    let mut buffer = BytesMut::with_capacity(HEADER_SIZE + payload.len());
    buffer.put_u8(TYPE_DATA);
    buffer.put_u64(exchange_id);
    buffer.extend_from_slice(payload);
    buffer.freeze()
}

/// Encode a compact ACK frame (no payload).
pub fn encode_ack(exchange_id: u64) -> Bytes {
    let mut buffer = BytesMut::with_capacity(HEADER_SIZE);
    buffer.put_u8(TYPE_ACK);
    buffer.put_u64(exchange_id);
    buffer.freeze()
}

/// Consumes a Bytes frame; reading the header does not copy the payload.
pub fn decode(mut buffer: Bytes) -> io::Result<Message> {
    if buffer.len() < HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too short",
        ));
    }

    let kind = buffer.get_u8();
    let exchange_id = buffer.get_u64();
    match kind {
        TYPE_DATA => Ok(Message::Data {
            exchange_id,
            payload: buffer,
        }),
        TYPE_ACK if buffer.is_empty() => Ok(Message::Ack { exchange_id }),
        TYPE_ACK => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ACK must have no payload",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unknown message type",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_roundtrip() {
        assert_eq!(
            decode(encode(42, b"Hello ConOps")).unwrap(),
            Message::Data {
                exchange_id: 42,
                payload: Bytes::from_static(b"Hello ConOps")
            },
        );
    }

    #[test]
    fn test_ack_roundtrip() {
        assert_eq!(
            decode(encode_ack(42)).unwrap(),
            Message::Ack { exchange_id: 42 }
        );
    }

    #[test]
    fn test_empty_payload() {
        assert_eq!(
            decode(encode(0, b"")).unwrap(),
            Message::Data {
                exchange_id: 0,
                payload: Bytes::new()
            },
        );
    }

    #[test]
    fn test_invalid_frames() {
        assert!(decode(Bytes::from_static(b"short")).is_err());
        let mut extra = encode_ack(10).to_vec();
        extra.push(0);
        assert!(decode(Bytes::from(extra)).is_err());
        let mut unknown = encode_ack(10).to_vec();
        unknown[0] = 255;
        assert!(decode(Bytes::from(unknown)).is_err());
    }
}
