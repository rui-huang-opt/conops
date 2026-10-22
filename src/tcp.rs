use std::io;

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const SMALL_FRAME_CAPACITY: usize = 512;
const LENGTH_PREFIX_SIZE: usize = 4;

pub async fn send_message<W>(writer: &mut W, data: &[u8], max_size: usize) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    if data.len() > max_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Message too large",
        ));
    }

    let len = u32::try_from(data.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Message length exceeds u32"))?;

    if data.len() <= SMALL_FRAME_CAPACITY - LENGTH_PREFIX_SIZE {
        // 小消息：栈上拼接，一次写入
        let mut buffer = [0u8; SMALL_FRAME_CAPACITY];

        buffer[..4].copy_from_slice(&len.to_be_bytes());
        buffer[4..4 + data.len()].copy_from_slice(data);

        writer.write_all(&buffer[..4 + data.len()]).await?;
    } else {
        // 大消息：原有路径，无额外完整复制
        writer.write_u32(len).await?;
        writer.write_all(data).await?;
    }

    Ok(())
}

pub async fn recv_message<R>(reader: &mut R, max_size: usize) -> io::Result<Bytes>
where
    R: AsyncRead + Unpin,
{
    let len = reader.read_u32().await? as usize;

    if len > max_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too large",
        ));
    }

    let mut buffer = BytesMut::with_capacity(len);
    let mut limited = reader.take(len as u64);

    while buffer.len() < len {
        if limited.read_buf(&mut buffer).await? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete message",
            ));
        }
    }

    Ok(buffer.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_send_recv() {
        let (mut tx, mut rx) = tokio::io::duplex(1024);
        let data = b"Hello ConOps";
        send_message(&mut tx, data, 1024).await.unwrap();
        let received = recv_message(&mut rx, 1024).await.unwrap();
        assert_eq!(received.as_ref(), data);
    }

    #[tokio::test]
    async fn test_message_too_large() {
        let (mut tx, _rx) = tokio::io::duplex(1024);
        assert!(send_message(&mut tx, &[0u8; 2048], 1024).await.is_err());
    }

    #[tokio::test]
    async fn test_empty_message() {
        let (mut tx, mut rx) = tokio::io::duplex(1024);
        send_message(&mut tx, b"", 1024).await.unwrap();
        assert!(recv_message(&mut rx, 1024).await.unwrap().is_empty());
    }
}
