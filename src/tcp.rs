use std::io::{self, IoSlice};

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

/// Write a length-prefixed TCP frame without concatenating the header and body.
///
/// On a vectored-capable socket, the common case submits both slices with one
/// write_vectored() call. A successful write can still be partial, so the
/// remaining header/body bytes must be written before returning.
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

    let header = len.to_be_bytes();
    let slices = [IoSlice::new(&header), IoSlice::new(data)];
    let written = loop {
        match writer.write_vectored(&slices).await {
            Ok(n) => break n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    };
    if written == 0 {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "Failed to write TCP frame",
        ));
    }

    if written < header.len() {
        // The initial write stopped inside the 4-byte prefix.
        writer.write_all(&header[written..]).await?;
        writer.write_all(data).await?;
    } else {
        // The prefix is complete; send only the unconsumed payload bytes.
        writer.write_all(&data[written - header.len()..]).await?;
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
    /// Records writes and simulates kernels accepting only a small prefix.
    struct LimitedWriter {
        output: Vec<u8>,
        max_per_write: usize,
        vectored_calls: usize,
        scalar_calls: usize,
    }

    impl LimitedWriter {
        fn new(max_per_write: usize) -> Self {
            Self {
                output: Vec::new(),
                max_per_write,
                vectored_calls: 0,
                scalar_calls: 0,
            }
        }
    }

    impl AsyncWrite for LimitedWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.scalar_calls += 1;
            let count = buf.len().min(self.max_per_write);
            self.output.extend_from_slice(&buf[..count]);
            Poll::Ready(Ok(count))
        }

        fn poll_write_vectored(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            bufs: &[IoSlice<'_>],
        ) -> Poll<io::Result<usize>> {
            self.vectored_calls += 1;
            let mut remaining = self.max_per_write;
            for buf in bufs {
                let count = buf.len().min(remaining);
                self.output.extend_from_slice(&buf[..count]);
                remaining -= count;
                if remaining == 0 {
                    break;
                }
            }
            Poll::Ready(Ok(self.max_per_write - remaining))
        }

        fn is_write_vectored(&self) -> bool {
            true
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn vectored_write_handles_partial_header_and_payload() {
        let payload = b"abcde";
        let mut expected = (payload.len() as u32).to_be_bytes().to_vec();
        expected.extend_from_slice(payload);

        for limit in [1, 2, 3, 4, 5, 6, 8, usize::MAX] {
            let mut writer = LimitedWriter::new(limit);
            send_message(&mut writer, payload, 64).await.unwrap();
            assert_eq!(writer.output, expected, "write limit: {limit}");
            assert_eq!(writer.vectored_calls, 1, "write limit: {limit}");
        }
    }

    #[tokio::test]
    async fn vectored_write_handles_empty_payload() {
        let mut writer = LimitedWriter::new(4);
        send_message(&mut writer, b"", 0).await.unwrap();
        assert_eq!(writer.output, [0, 0, 0, 0]);
        assert_eq!(writer.vectored_calls, 1);
        assert_eq!(writer.scalar_calls, 0);
    }

    #[tokio::test]
    async fn vectored_write_zero_is_an_error() {
        let mut writer = LimitedWriter::new(0);
        let err = send_message(&mut writer, b"xyz", 32).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WriteZero);
    }

    #[tokio::test]
    async fn vectored_write_large_payload_without_frame_concatenation() {
        let payload = vec![23u8; 256 * 1024];
        let mut writer = LimitedWriter::new(usize::MAX);
        send_message(&mut writer, &payload, payload.len())
            .await
            .unwrap();
        assert_eq!(&writer.output[..4], &(payload.len() as u32).to_be_bytes());
        assert_eq!(&writer.output[4..], payload.as_slice());
        assert_eq!(writer.vectored_calls, 1);
        assert_eq!(writer.scalar_calls, 0);
    }
}
