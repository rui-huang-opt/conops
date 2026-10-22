use std::io;

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

pub const RECEIVE_BUFFER_SIZE: usize = 8 * 1024;

/// One of these is owned by each reader task for the whole TCP session.
/// Never create/drop a new buffered reader between messages: it may have
/// already consumed bytes belonging to the next message from the socket.
pub struct FrameReader<R> {
    reader: BufReader<R>,
}

impl<R: AsyncRead + Unpin> FrameReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(RECEIVE_BUFFER_SIZE, reader),
        }
    }

    pub async fn recv(&mut self, max_size: usize) -> io::Result<Bytes> {
        recv_message(&mut self.reader, max_size).await
    }
}

// The wire format and DATA send path are deliberately unchanged.
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
    writer.write_u32(len).await?;
    writer.write_all(data).await?;
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
    use std::{
        pin::Pin,
        task::{Context, Poll},
    };
    use tokio::io::ReadBuf;

    #[tokio::test]
    async fn send_recv() {
        let (mut tx, mut rx) = tokio::io::duplex(1024);
        send_message(&mut tx, b"Hello ConOps", 1024).await.unwrap();
        assert_eq!(
            recv_message(&mut rx, 1024).await.unwrap().as_ref(),
            b"Hello ConOps"
        );
    }

    #[tokio::test]
    async fn message_too_large() {
        let (mut tx, _rx) = tokio::io::duplex(1024);
        assert!(send_message(&mut tx, &[0u8; 2048], 1024).await.is_err());
    }

    #[tokio::test]
    async fn persistent_reader_preserves_prefetched_frames() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        for data in [b"a".as_slice(), b"", b"second", b"last"] {
            send_message(&mut tx, data, 1024).await.unwrap();
        }
        let mut r = FrameReader::new(rx);
        for data in [b"a".as_slice(), b"", b"second", b"last"] {
            assert_eq!(r.recv(1024).await.unwrap().as_ref(), data);
        }
    }

    #[tokio::test]
    async fn large_frame_followed_by_small_frame() {
        let (mut tx, rx) = tokio::io::duplex(128);
        let data = vec![42; 128 * 1024];
        let writer_data = data.clone();
        let writer = tokio::spawn(async move {
            send_message(&mut tx, &writer_data, 256 * 1024)
                .await
                .unwrap();
            send_message(&mut tx, b"tail", 1024).await.unwrap();
        });
        let mut r = FrameReader::new(rx);
        let retained = r.recv(256 * 1024).await.unwrap();
        assert_eq!(r.recv(1024).await.unwrap().as_ref(), b"tail");
        assert_eq!(retained.as_ref(), data.as_slice());
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn truncated_body_and_oversized_header() {
        let (mut tx, rx) = tokio::io::duplex(64);
        tx.write_u32(10).await.unwrap();
        tx.write_all(b"abc").await.unwrap();
        drop(tx);
        assert_eq!(
            FrameReader::new(rx).recv(64).await.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        let (mut tx, rx) = tokio::io::duplex(64);
        tx.write_u32(65).await.unwrap();
        assert_eq!(
            FrameReader::new(rx).recv(64).await.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    struct Fragmented {
        data: Vec<u8>,
        position: usize,
    }
    impl AsyncRead for Fragmented {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            dst: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            if self.position < self.data.len() && dst.remaining() > 0 {
                let byte = self.data[self.position];
                self.position += 1;
                dst.put_slice(&[byte]);
            }
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn single_byte_fragmentation() {
        let raw = [0, 0, 0, 3, b'a', b'b', b'c', 0, 0, 0, 1, b'd'].to_vec();
        let mut r = FrameReader::new(Fragmented {
            data: raw,
            position: 0,
        });
        assert_eq!(r.recv(64).await.unwrap().as_ref(), b"abc");
        assert_eq!(r.recv(64).await.unwrap().as_ref(), b"d");
    }
}
