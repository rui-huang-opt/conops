use std::collections::{HashMap, HashSet};

use bytes::Bytes;

/// Collects exactly one message from each peer for an exchange.
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

/// Messages for exchange IDs that the local node has not started yet.
/// Both message count and retained payload bytes are bounded.
#[derive(Debug)]
pub struct EarlyMessages {
    messages: HashMap<u64, HashMap<String, Bytes>>,
    count: usize,
    bytes: usize,
    max_messages: usize,
    max_bytes: usize,
}

impl EarlyMessages {
    pub fn new(max_messages: usize, max_bytes: usize) -> Self {
        Self {
            messages: HashMap::new(),
            count: 0,
            bytes: 0,
            max_messages,
            max_bytes,
        }
    }

    pub fn insert(
        &mut self,
        exchange_id: u64,
        peer: String,
        payload: Bytes,
    ) -> Result<bool, &'static str> {
        if self
            .messages
            .get(&exchange_id)
            .is_some_and(|messages| messages.contains_key(&peer))
        {
            return Ok(false);
        }

        if self.count >= self.max_messages
            || payload.len() > self.max_bytes.saturating_sub(self.bytes)
        {
            return Err("Early message buffer full");
        }

        self.count += 1;
        self.bytes += payload.len();
        self.messages
            .entry(exchange_id)
            .or_default()
            .insert(peer, payload);
        Ok(true)
    }

    pub fn take(&mut self, exchange_id: u64) -> HashMap<String, Bytes> {
        let messages = self.messages.remove(&exchange_id).unwrap_or_default();
        self.count -= messages.len();
        self.bytes -= messages.values().map(Bytes::len).sum::<usize>();
        messages
    }

    /// Drop all cached messages through `exchange_id` (inclusive).
    pub fn discard_through(&mut self, exchange_id: u64) {
        let expired: Vec<u64> = self
            .messages
            .keys()
            .copied()
            .filter(|id| *id <= exchange_id)
            .collect();

        for id in expired {
            let _ = self.take(id);
        }
    }

    /// Remove cached data from a disconnected peer (old sessions must not leak).
    pub fn remove_peer(&mut self, peer_id: &str) {
        let ids: Vec<u64> = self.messages.keys().copied().collect();
        for id in ids {
            let mut empty = false;
            if let Some(messages) = self.messages.get_mut(&id) {
                if let Some(data) = messages.remove(peer_id) {
                    self.count -= 1;
                    self.bytes -= data.len();
                }
                empty = messages.is_empty();
            }
            if empty {
                self.messages.remove(&id);
            }
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
        assert_eq!(
            batch.insert("B".into(), Bytes::from_static(&[1, 2, 3])),
            Ok(true)
        );
        assert_eq!(
            batch.insert("B".into(), Bytes::from_static(&[4, 5, 6])),
            Ok(false)
        );
        assert!(!batch.is_complete());
        assert_eq!(
            batch.insert("C".into(), Bytes::from_static(&[7, 8, 9])),
            Ok(true)
        );
        assert!(batch.is_complete());
        assert_eq!(batch.received["B"].as_ref(), &[1, 2, 3]);
        assert!(
            batch
                .insert("D".into(), Bytes::from_static(b"bad"))
                .is_err()
        );
    }

    #[test]
    fn test_take() {
        let mut batch = Batch::new(10, vec!["B".into(), "C".into()]);
        batch
            .insert("B".into(), Bytes::from_static(b"Hello B"))
            .unwrap();
        let mut batch = batch.take().unwrap_err();
        assert!(!batch.is_complete());
        batch
            .insert("C".into(), Bytes::from_static(b"Hello C"))
            .unwrap();
        let received = batch.take().unwrap();
        assert_eq!(received["B"].as_ref(), b"Hello B");
        assert_eq!(received["C"].as_ref(), b"Hello C");
    }

    #[test]
    fn test_empty_batch() {
        let batch = Batch::new(0, vec![]);
        assert!(batch.is_complete());
        assert!(batch.take().unwrap().is_empty());
    }

    #[test]
    fn test_early_messages() {
        let mut early = EarlyMessages::new(10, 1024);
        assert_eq!(
            early.insert(10, "A".into(), Bytes::from_static(b"A10")),
            Ok(true)
        );
        assert_eq!(
            early.insert(11, "A".into(), Bytes::from_static(b"A11")),
            Ok(true)
        );
        assert_eq!(
            early.insert(10, "C".into(), Bytes::from_static(b"C10")),
            Ok(true)
        );
        assert_eq!(
            early.insert(10, "A".into(), Bytes::from_static(b"duplicate")),
            Ok(false)
        );

        let messages = early.take(10);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages["A"].as_ref(), b"A10");
        assert_eq!(messages["C"].as_ref(), b"C10");
        assert_eq!(early.take(11)["A"].as_ref(), b"A11");
        assert!(early.take(11).is_empty());
    }

    #[test]
    fn test_early_limits_and_reclaim() {
        let mut early = EarlyMessages::new(2, 5);
        assert_eq!(
            early.insert(10, "A".into(), Bytes::from_static(b"123")),
            Ok(true)
        );
        assert!(
            early
                .insert(10, "B".into(), Bytes::from_static(b"456"))
                .is_err()
        );
        assert_eq!(
            early.insert(11, "B".into(), Bytes::from_static(b"4")),
            Ok(true)
        );
        assert!(early.insert(12, "C".into(), Bytes::new()).is_err());
        early.discard_through(10);
        assert_eq!(
            early.insert(12, "C".into(), Bytes::from_static(b"5678")),
            Ok(true)
        );
        early.remove_peer("B");
        assert!(early.take(11).is_empty());
        assert_eq!(early.take(12)["C"].as_ref(), b"5678");
        assert_eq!(
            early.insert(13, "Z".into(), Bytes::from_static(b"12345")),
            Ok(true)
        );
    }
}
