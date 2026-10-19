use std::collections::{HashMap, HashSet};

use bytes::Bytes;

#[derive(Debug)]
pub struct Batch {
    exchange_id: u64,
    expected: HashSet<String>,
    received: HashMap<String, Bytes>,
}

impl Batch {
    pub fn new(exchange_id: u64, peers: Vec<String>) -> Self {
        Self {
            exchange_id,
            expected: peers.into_iter().collect(),
            received: HashMap::new(),
        }
    }

    pub fn exchange_id(&self) -> u64 {
        self.exchange_id
    }

    pub fn insert(&mut self, peer: String, payload: Bytes) -> Result<bool, &'static str> {
        if !self.expected.contains(&peer) {
            return Err("Unexpected peer");
        }

        if self.received.contains_key(&peer) {
            return Ok(false);
        }

        self.received.insert(peer, payload);
        Ok(true)
    }

    pub fn is_complete(&self) -> bool {
        self.received.len() == self.expected.len()
    }

    pub fn take(self) -> Result<HashMap<String, Bytes>, Self> {
        if self.is_complete() {
            Ok(self.received)
        } else {
            Err(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let mut batch = Batch::new(10, vec!["B".into(), "C".into()]);

        assert_eq!(batch.exchange_id(), 10);
        assert!(!batch.is_complete());

        // 收到 B
        assert_eq!(
            batch.insert("B".into(), Bytes::from_static(&[1, 2, 3])),
            Ok(true),
        );

        assert!(!batch.is_complete());

        // B 重复发送，不应该覆盖原来的数据
        assert_eq!(
            batch.insert("B".into(), Bytes::from_static(&[4, 5, 6])),
            Ok(false),
        );

        assert!(!batch.is_complete());

        // 收到 C
        assert_eq!(
            batch.insert("C".into(), Bytes::from_static(&[7, 8, 9])),
            Ok(true),
        );

        assert!(batch.is_complete());

        // 验证重复消息没有覆盖最初的数据
        assert_eq!(batch.received.get("B").unwrap().as_ref(), &[1, 2, 3],);
    }

    #[test]
    fn test_take() {
        let mut batch = Batch::new(10, vec!["B".into(), "C".into()]);

        // 只收到 B
        batch
            .insert("B".into(), Bytes::from_static(b"Hello B"))
            .unwrap();

        // 未收齐，take 失败，但返回原来的 Batch
        let mut batch = batch.take().unwrap_err();

        assert!(!batch.is_complete());

        // 继续收到 C
        batch
            .insert("C".into(), Bytes::from_static(b"Hello C"))
            .unwrap();

        assert!(batch.is_complete());

        // 现在可以取走所有数据
        let received = batch.take().unwrap();

        assert_eq!(received.len(), 2);
        assert_eq!(received["B"].as_ref(), b"Hello B");
        assert_eq!(received["C"].as_ref(), b"Hello C");

        // 此时 batch 已经被移动，不能再使用
    }
}
