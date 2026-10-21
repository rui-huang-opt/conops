use std::collections::{HashMap, HashSet, VecDeque};

use bytes::Bytes;

/// Collects exactly one message from each peer for an exchange.
#[derive(Debug)]
pub struct Batch {
    key: Bytes,
    expected: HashSet<String>,
    received: HashMap<String, Bytes>,
}

impl Batch {
    pub fn new(key: Bytes, peers: Vec<String>) -> Self {
        Self {
            key,
            expected: peers.into_iter().collect(),
            received: HashMap::new(),
        }
    }

    pub fn key(&self) -> &Bytes {
        &self.key
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

/// Messages for exchange keys not yet started locally.
/// In sequential exchange programs a peer can be one exchange ahead.
#[derive(Debug)]
pub struct EarlyMessages {
    messages: HashMap<Bytes, HashMap<String, Bytes>>,
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
        key: Bytes,
        peer: String,
        payload: Bytes,
    ) -> Result<bool, &'static str> {
        if self
            .messages
            .get(&key)
            .is_some_and(|messages| messages.contains_key(&peer))
        {
            return Ok(false);
        }

        // A single active Exchange per node means a correctly ordered peer
        // should never have more than one *future* DATA outstanding here.
        if self
            .messages
            .values()
            .any(|messages| messages.contains_key(&peer))
        {
            return Err("Peer already has an early exchange message");
        }

        if self.count >= self.max_messages
            || payload.len() > self.max_bytes.saturating_sub(self.bytes)
        {
            return Err("Early message buffer full");
        }

        self.count += 1;
        self.bytes += payload.len();
        self.messages.entry(key).or_default().insert(peer, payload);
        Ok(true)
    }

    pub fn take(&mut self, key: &Bytes) -> HashMap<String, Bytes> {
        let messages = self.messages.remove(key).unwrap_or_default();
        self.count -= messages.len();
        self.bytes -= messages.values().map(Bytes::len).sum::<usize>();
        messages
    }

    /// Remove cached data from a disconnected peer if explicitly requested.
    pub fn remove_peer(&mut self, peer_id: &str) {
        let keys: Vec<Bytes> = self.messages.keys().cloned().collect();
        for key in keys {
            let mut empty = false;
            if let Some(messages) = self.messages.get_mut(&key) {
                if let Some(data) = messages.remove(peer_id) {
                    self.count -= 1;
                    self.bytes -= data.len();
                }
                empty = messages.is_empty();
            }
            if empty {
                self.messages.remove(&key);
            }
        }
    }
}

/// Bounded record of completed or cancelled keys; stores keys, not payloads.
#[derive(Debug)]
pub struct CompletedKeys {
    keys: HashSet<Bytes>,
    order: VecDeque<Bytes>,
    capacity: usize,
}

impl CompletedKeys {
    pub fn new(capacity: usize) -> Self {
        Self {
            keys: HashSet::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn contains(&self, key: &Bytes) -> bool {
        self.keys.contains(key)
    }

    pub fn insert(&mut self, key: Bytes) {
        if self.capacity == 0 || !self.keys.insert(key.clone()) {
            return;
        }
        self.order.push_back(key);
        if self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.keys.remove(&oldest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(text: &'static [u8]) -> Bytes {
        Bytes::from_static(text)
    }

    #[test]
    fn test_batch() {
        let mut batch = Batch::new(key(b"y"), vec!["B".into(), "C".into()]);
        assert_eq!(batch.key(), &key(b"y"));
        assert!(!batch.is_complete());
        assert_eq!(batch.insert("B".into(), key(b"123")), Ok(true));
        assert_eq!(batch.insert("B".into(), key(b"456")), Ok(false));
        assert_eq!(batch.insert("C".into(), key(b"789")), Ok(true));
        assert!(batch.is_complete());
        assert_eq!(batch.received["B"].as_ref(), b"123");
        assert_eq!(batch.received["C"].as_ref(), b"789");
        assert!(batch.insert("D".into(), key(b"bad")).is_err());
    }

    #[test]
    fn test_take() {
        let mut batch = Batch::new(key(b"x"), vec!["B".into(), "C".into()]);
        batch.insert("B".into(), key(b"Hello B")).unwrap();
        let mut batch = batch.take().unwrap_err();
        batch.insert("C".into(), key(b"Hello C")).unwrap();
        let received = batch.take().unwrap();
        assert_eq!(received["B"].as_ref(), b"Hello B");
        assert_eq!(received["C"].as_ref(), b"Hello C");
    }

    #[test]
    fn test_empty_batch() {
        let batch = Batch::new(key(b"x"), vec![]);
        assert!(batch.is_complete());
        assert!(batch.take().unwrap().is_empty());
    }

    #[test]
    fn test_early_messages_unordered_keys() {
        let mut early = EarlyMessages::new(10, 1024);
        assert_eq!(early.insert(key(b"z"), "A".into(), key(b"A")), Ok(true));
        assert_eq!(early.insert(key(b"a"), "B".into(), key(b"B")), Ok(true));
        assert_eq!(early.insert(key(b"z"), "C".into(), key(b"C")), Ok(true));
        assert_eq!(
            early.insert(key(b"z"), "A".into(), key(b"duplicate")),
            Ok(false)
        );
        // One early DATA per peer: other future keys cannot accumulate.
        assert!(early.insert(key(b"next"), "A".into(), key(b"bad")).is_err());
        let messages = early.take(&key(b"z"));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages["A"].as_ref(), b"A");
        assert_eq!(messages["C"].as_ref(), b"C");
        assert_eq!(early.take(&key(b"a"))["B"].as_ref(), b"B");
    }

    #[test]
    fn test_early_limits_and_reclaim() {
        let mut early = EarlyMessages::new(2, 5);
        assert_eq!(early.insert(key(b"z"), "A".into(), key(b"123")), Ok(true));
        assert!(early.insert(key(b"z"), "B".into(), key(b"456")).is_err());
        assert_eq!(early.insert(key(b"a"), "B".into(), key(b"4")), Ok(true));
        assert!(early.insert(key(b"c"), "C".into(), Bytes::new()).is_err());
        let _ = early.take(&key(b"z"));
        assert_eq!(early.insert(key(b"c"), "C".into(), key(b"5678")), Ok(true));
        early.remove_peer("B");
        assert!(early.take(&key(b"a")).is_empty());
        assert_eq!(early.take(&key(b"c"))["C"].as_ref(), b"5678");
    }

    #[test]
    fn test_completed_keys_are_bounded() {
        let mut history = CompletedKeys::new(2);
        history.insert(key(b"z"));
        history.insert(key(b"a"));
        history.insert(key(b"z")); // duplicate does not change eviction order
        assert!(history.contains(&key(b"z")));
        history.insert(key(b"q"));
        assert!(!history.contains(&key(b"z")));
        assert!(history.contains(&key(b"a")));
        assert!(history.contains(&key(b"q")));
    }
}
