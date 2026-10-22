use std::io;

use bytes::{Buf, BufMut, Bytes, BytesMut};

// The outer TCP frame still uses a 4-byte big-endian length prefix.
// Experimental inner frame: type (1 byte) | key length (2 bytes) | key | payload.
// TYPE_ACK is deliberately unsupported in this benchmark-only protocol.
const TYPE_DATA: u8 = 0;
pub const HEADER_SIZE: usize = 3;
pub const MAX_KEY_SIZE: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub enum Message {
    Data { key: Bytes, payload: Bytes },
}

fn key_length(key: &[u8]) -> io::Result<u16> {
    if key.is_empty() || key.len() > MAX_KEY_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid exchange key length",
        ));
    }
    Ok(key.len() as u16)
}

/// Encode a NumPy state in a single allocation.
/// DATA payload: 4-byte metadata length | metadata | contiguous array bytes.
pub fn encode_state(key: &[u8], meta: &[u8], payload: &[u8]) -> io::Result<Bytes> {
    let key_len = key_length(key)?;
    let meta_len = u32::try_from(meta.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Metadata exceeds u32"))?;
    let capacity = HEADER_SIZE
        .checked_add(key.len())
        .and_then(|size| size.checked_add(4))
        .and_then(|size| size.checked_add(meta.len()))
        .and_then(|size| size.checked_add(payload.len()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Message too large"))?;

    let mut buffer = BytesMut::with_capacity(capacity);
    buffer.put_u8(TYPE_DATA);
    buffer.put_u16(key_len);
    buffer.extend_from_slice(key);
    buffer.put_u32(meta_len);
    buffer.extend_from_slice(meta);
    buffer.extend_from_slice(payload);
    Ok(buffer.freeze())
}

/// Consumes an owned DATA frame. Copy the key only, not the large payload.
/// Copying the key avoids pinning the payload allocation in key history.
pub fn decode(mut buffer: Bytes) -> io::Result<Message> {
    if buffer.len() < HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too short",
        ));
    }

    let kind = buffer.get_u8();
    if kind != TYPE_DATA {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unknown message type",
        ));
    }

    let key_len = buffer.get_u16() as usize;
    if key_len == 0 || key_len > MAX_KEY_SIZE || buffer.len() < key_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid exchange key length",
        ));
    }
    let key = Bytes::copy_from_slice(&buffer[..key_len]);
    buffer.advance(key_len);
    Ok(Message::Data {
        key,
        payload: buffer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_roundtrip() {
        let key = b"\x92\x02\xa1x"; // msgpack: (2, "x")
        let meta = b"meta";
        let payload = b"\x00\x01\xff";
        let encoded = encode_state(key, meta, payload).unwrap();

        let mut expected = Vec::new();
        expected.push(TYPE_DATA);
        expected.extend_from_slice(&(key.len() as u16).to_be_bytes());
        expected.extend_from_slice(key);
        expected.extend_from_slice(&(meta.len() as u32).to_be_bytes());
        expected.extend_from_slice(meta);
        expected.extend_from_slice(payload);
        assert_eq!(encoded.as_ref(), expected.as_slice());

        let mut state = Vec::new();
        state.extend_from_slice(&(meta.len() as u32).to_be_bytes());
        state.extend_from_slice(meta);
        state.extend_from_slice(payload);
        assert_eq!(
            decode(encoded).unwrap(),
            Message::Data {
                key: Bytes::copy_from_slice(key),
                payload: Bytes::from(state),
            }
        );
    }

    #[test]
    fn test_empty_payload() {
        let encoded = encode_state(b"x", b"", b"").unwrap();
        assert_eq!(encoded.len(), HEADER_SIZE + 1 + 4);
        assert_eq!(
            decode(encoded).unwrap(),
            Message::Data {
                key: Bytes::from_static(b"x"),
                payload: Bytes::from_static(&[0, 0, 0, 0]),
            }
        );
    }

    #[test]
    fn test_data_key_does_not_pin_payload_allocation() {
        let encoded = encode_state(b"key", b"", &vec![0u8; 1024]).unwrap();
        let original_key_ptr = encoded[HEADER_SIZE..].as_ptr();
        match decode(encoded).unwrap() {
            Message::Data { key, .. } => assert_ne!(key.as_ptr(), original_key_ptr),
        }
    }

    #[test]
    fn test_invalid_frames() {
        assert!(decode(Bytes::from_static(b"x")).is_err());
        assert!(encode_state(b"", b"", b"").is_err());
        assert!(encode_state(&vec![0; MAX_KEY_SIZE + 1], b"", b"").is_err());

        let mut unknown = encode_state(b"x", b"", b"").unwrap().to_vec();
        unknown[0] = 1; // The previous ACK type is invalid in this experiment.
        assert!(decode(Bytes::from(unknown)).is_err());

        let mut truncated = encode_state(b"x", b"", b"").unwrap().to_vec();
        truncated[2] = 20;
        assert!(decode(Bytes::from(truncated)).is_err());

        let mut zero_key = encode_state(b"x", b"", b"").unwrap().to_vec();
        zero_key[2] = 0;
        assert!(decode(Bytes::from(zero_key)).is_err());
    }
}
