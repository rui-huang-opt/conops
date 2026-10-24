use std::io;

use bytes::{Buf, BufMut, Bytes, BytesMut};

// Unchanged wire format:
// outer: u32 big-endian length
// inner: type (u8) | key length (u16 BE) | key | optional DATA payload
const TYPE_DATA: u8 = 0;
const TYPE_ACK: u8 = 1;
pub const HEADER_SIZE: usize = 3;
pub const MAX_KEY_SIZE: usize = 256;
pub const ACK_BUFFER_SIZE: usize = HEADER_SIZE + MAX_KEY_SIZE;

#[derive(Debug, PartialEq, Eq)]
pub enum Message {
    Data { key: Bytes, payload: Bytes },
    Ack { key: Bytes },
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

/// One allocation for broadcast; Bytes clones share the allocation.
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

/// The writer owns this scratch buffer for its entire lifetime. No ACK-body
/// allocation is required, and no scratch reference escapes a completed write.
pub fn encode_ack_into<'a>(
    key: &[u8],
    output: &'a mut [u8; ACK_BUFFER_SIZE],
) -> io::Result<&'a [u8]> {
    let key_len = key_length(key)?;
    output[0] = TYPE_ACK;
    output[1..3].copy_from_slice(&key_len.to_be_bytes());
    output[3..3 + key.len()].copy_from_slice(key);
    Ok(&output[..3 + key.len()])
}

/// Kept for Rust callers/tests; the networking hot path uses encode_ack_into.
pub fn encode_ack(key: &[u8]) -> Bytes {
    let mut buffer = [0; ACK_BUFFER_SIZE];
    Bytes::copy_from_slice(encode_ack_into(key, &mut buffer).expect("validated exchange key"))
}

/// Copies DATA keys to avoid pinning a large payload in completed-key history.
/// The DATA payload itself remains a slice of the received Bytes allocation.
pub fn decode(mut buffer: Bytes) -> io::Result<Message> {
    if buffer.len() < HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too short",
        ));
    }
    let kind = buffer.get_u8();
    let key_len = buffer.get_u16() as usize;
    if key_len == 0 || key_len > MAX_KEY_SIZE || buffer.len() < key_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid exchange key length",
        ));
    }
    match kind {
        TYPE_DATA => {
            // Do not acknowledge a structurally truncated DATA frame.
            let rest = &buffer[key_len..];
            if rest.len() < 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Missing DATA metadata length",
                ));
            }
            let meta_len = u32::from_be_bytes(rest[..4].try_into().expect("four bytes")) as usize;
            if meta_len > rest.len() - 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Truncated DATA metadata",
                ));
            }
            let key = Bytes::copy_from_slice(&buffer[..key_len]);
            buffer.advance(key_len);
            Ok(Message::Data {
                key,
                payload: buffer,
            })
        }
        TYPE_ACK if buffer.len() == key_len => Ok(Message::Ack { key: buffer }),
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
    fn data_roundtrip_and_wire_compatibility() {
        let key = b"\x92\x02\xa1x";
        let meta = b"meta";
        let payload = b"\x00\x01\xff";
        let encoded = encode_state(key, meta, payload).unwrap();
        let mut expected = vec![TYPE_DATA];
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
    fn ack_scratch_roundtrip_and_maximum_length() {
        let mut scratch = [0u8; ACK_BUFFER_SIZE];
        for len in [1, 8, MAX_KEY_SIZE] {
            let key = vec![b'x'; len];
            let encoded = encode_ack_into(&key, &mut scratch).unwrap();
            assert_eq!(encoded, encode_ack(&key).as_ref());
            assert_eq!(
                decode(Bytes::copy_from_slice(encoded)).unwrap(),
                Message::Ack {
                    key: Bytes::from(key)
                }
            );
        }
        assert!(encode_ack_into(b"", &mut scratch).is_err());
        assert!(encode_ack_into(&[0; MAX_KEY_SIZE + 1], &mut scratch).is_err());
    }

    #[test]
    fn empty_payload_is_legal() {
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
    fn data_key_does_not_pin_payload_allocation() {
        let encoded = encode_state(b"key", b"", &vec![0; 1024]).unwrap();
        let original_key = encoded[HEADER_SIZE..].as_ptr();
        let original_payload = encoded[HEADER_SIZE + 3..].as_ptr();
        match decode(encoded).unwrap() {
            Message::Data { key, payload } => {
                assert_ne!(key.as_ptr(), original_key);
                assert_eq!(payload.as_ptr(), original_payload);
            }
            _ => panic!("expected DATA"),
        }
    }

    #[test]
    fn invalid_frames() {
        assert!(decode(Bytes::from_static(b"x")).is_err());
        assert!(encode_state(b"", b"", b"").is_err());
        assert!(encode_state(&[0; MAX_KEY_SIZE + 1], b"", b"").is_err());
        let mut extra = encode_ack(b"x").to_vec();
        extra.push(0);
        assert!(decode(Bytes::from(extra)).is_err());
        let mut unknown = encode_ack(b"x").to_vec();
        unknown[0] = 255;
        assert!(decode(Bytes::from(unknown)).is_err());
        let mut truncated = encode_ack(b"x").to_vec();
        truncated[2] = 20;
        assert!(decode(Bytes::from(truncated)).is_err());
        let mut zero = encode_ack(b"x").to_vec();
        zero[2] = 0;
        assert!(decode(Bytes::from(zero)).is_err());
        assert!(decode(Bytes::from_static(&[0, 0, 1, b'x'])).is_err());
        assert!(decode(Bytes::from_static(&[0, 0, 1, b'x', 0, 0, 0, 1])).is_err());
    }
}
