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

/// Encode the NumPy adapter's state directly into a single DATA allocation.
/// DATA payload: 4-byte metadata length | metadata | contiguous array bytes.
pub fn encode_state(exchange_id: u64, meta: &[u8], payload: &[u8]) -> io::Result<Bytes> {
    let meta_len = u32::try_from(meta.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Metadata exceeds u32"))?;
    let capacity = HEADER_SIZE
        .checked_add(4)
        .and_then(|size| size.checked_add(meta.len()))
        .and_then(|size| size.checked_add(payload.len()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Message too large"))?;

    let mut buffer = BytesMut::with_capacity(capacity);
    buffer.put_u8(TYPE_DATA);
    buffer.put_u64(exchange_id);
    buffer.put_u32(meta_len);
    buffer.extend_from_slice(meta);
    buffer.extend_from_slice(payload);
    Ok(buffer.freeze())
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
        let meta = b"meta";
        let payload = b"\x00\x01\xff";

        let encoded = encode_state(42, meta, payload).unwrap();

        let mut expected = Vec::new();
        expected.push(TYPE_DATA);
        expected.extend_from_slice(&42u64.to_be_bytes());
        expected.extend_from_slice(&(meta.len() as u32).to_be_bytes());
        expected.extend_from_slice(meta);
        expected.extend_from_slice(payload);

        assert_eq!(encoded.as_ref(), expected.as_slice());

        assert_eq!(
            decode(encoded).unwrap(),
            Message::Data {
                exchange_id: 42,
                payload: Bytes::from(expected[HEADER_SIZE..].to_vec()),
            }
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
        let encoded = encode_state(0, b"", b"").unwrap();

        assert_eq!(encoded.len(), HEADER_SIZE + 4);

        assert_eq!(
            decode(encoded).unwrap(),
            Message::Data {
                exchange_id: 0,
                payload: Bytes::from_static(&[0, 0, 0, 0]),
            }
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
