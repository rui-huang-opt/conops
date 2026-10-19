use std::io;

pub fn encode(exchange_id: u64, payload: &[u8]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(8 + payload.len());

    buffer.extend_from_slice(&exchange_id.to_be_bytes());

    buffer.extend_from_slice(payload);

    buffer
}

pub fn decode(buffer: &[u8]) -> io::Result<(u64, &[u8])> {
    if buffer.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too short",
        ));
    }

    let id_bytes: [u8; 8] = buffer[..8]
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Exchange ID"))?;

    let exchange_id = u64::from_be_bytes(id_bytes);

    let payload = &buffer[8..];

    Ok((exchange_id, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let exchange_id = 42;
        let payload = b"Hello ConOps";

        let encoded = encode(exchange_id, payload);
        let (decoded_id, decoded_payload) = decode(&encoded).unwrap();

        assert_eq!(decoded_id, exchange_id);
        assert_eq!(decoded_payload, payload);
    }
}
