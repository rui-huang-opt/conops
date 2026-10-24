use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use bytes::Bytes;

/// Stable for the lifetime of a Network, including across TCP reconnections.
/// Indices are local to a Network and are never sent on the wire.
pub type PeerIndex = usize;

/// Immutable topology snapshot. Fixed-topology networks create this only once;
/// manual-mode networks rebuild it only when their connected peer set changes.
#[derive(Debug)]
pub struct PeerLayout {
    peers: Vec<(PeerIndex, String)>,
    slots: Vec<Option<usize>>,
}

impl PeerLayout {
    pub fn new(mut peers: Vec<(PeerIndex, String)>) -> Self {
        peers.sort_unstable_by_key(|(index, _)| *index);
        let capacity = peers.last().map_or(0, |(index, _)| index + 1);
        let mut slots = vec![None; capacity];
        for (slot, (index, _)) in peers.iter().enumerate() {
            assert!(
                slots[*index].replace(slot).is_none(),
                "Duplicate peer index"
            );
        }
        Self { peers, slots }
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn slot(&self, index: PeerIndex) -> Option<usize> {
        self.slots.get(index).copied().flatten()
    }

    pub fn peers(&self) -> &[(PeerIndex, String)] {
        &self.peers
    }
}

/// One payload per neighbor, without rebuilding a String HashSet/HashMap for
/// every exchange. Payloads are moved, not copied; slot storage can be recycled.
#[derive(Debug)]
pub struct Batch {
    key: Bytes,
    layout: Arc<PeerLayout>,
    received: Vec<Option<Bytes>>,
    remaining: usize,
}

impl Batch {
    pub fn new(key: Bytes, layout: Arc<PeerLayout>, mut recycled: Vec<Option<Bytes>>) -> Self {
        recycled.clear();
        recycled.resize_with(layout.len(), || None);
        let remaining = layout.len();
        Self {
            key,
            layout,
            received: recycled,
            remaining,
        }
    }

    pub fn key(&self) -> &Bytes {
        &self.key
    }

    pub fn expects(&self, peer: PeerIndex) -> bool {
        self.layout.slot(peer).is_some()
    }

    pub fn insert(&mut self, peer: PeerIndex, payload: Bytes) -> Result<bool, &'static str> {
        let slot = self.layout.slot(peer).ok_or("Unexpected peer")?;
        if self.received[slot].is_some() {
            return Ok(false);
        }
        self.received[slot] = Some(payload);
        self.remaining -= 1;
        Ok(true)
    }

    pub fn is_complete(&self) -> bool {
        self.remaining == 0
    }

    /// String keys are materialized only at the Python-facing result boundary.
    /// The returned slot vector has retained capacity and contains no payloads.
    pub fn take(mut self) -> Result<(HashMap<String, Bytes>, Vec<Option<Bytes>>), Self> {
        if !self.is_complete() {
            return Err(self);
        }
        let mut result = HashMap::with_capacity(self.layout.len());
        for (slot, (_, name)) in self.layout.peers().iter().enumerate() {
            result.insert(
                name.clone(),
                self.received[slot].take().expect("complete batch"),
            );
        }
        self.received.clear();
        Ok((result, self.received))
    }

    pub fn recycle(mut self) -> Vec<Option<Bytes>> {
        self.received.clear();
        self.received
    }
}

#[derive(Debug)]
struct EarlyEntry {
    key: Bytes,
    payload: Bytes,
}

/// One future DATA per peer, as in the original sequential-exchange contract.
/// Storage is indexed by stable peer index, not nested key/String hash maps.
#[derive(Debug)]
pub struct EarlyMessages {
    entries: Vec<Option<EarlyEntry>>,
    count: usize,
    bytes: usize,
    max_messages: usize,
    max_bytes: usize,
}

impl EarlyMessages {
    pub fn new(max_messages: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            count: 0,
            bytes: 0,
            max_messages,
            max_bytes,
        }
    }

    /// Called on registration, never with an index supplied by a remote peer.
    pub fn reserve_peers(&mut self, count: usize) {
        if self.entries.len() < count {
            self.entries.resize_with(count, || None);
        }
    }

    pub fn insert(
        &mut self,
        key: Bytes,
        peer: PeerIndex,
        payload: Bytes,
    ) -> Result<bool, &'static str> {
        if let Some(Some(entry)) = self.entries.get(peer) {
            return if entry.key == key {
                Ok(false)
            } else {
                Err("Peer already has an early exchange message")
            };
        }
        if self.count >= self.max_messages
            || payload.len() > self.max_bytes.saturating_sub(self.bytes)
        {
            return Err("Early message buffer full");
        }
        self.reserve_peers(peer + 1);
        self.count += 1;
        self.bytes += payload.len();
        self.entries[peer] = Some(EarlyEntry { key, payload });
        Ok(true)
    }

    /// Non-destructive validation. This must precede any early-data removal.
    pub fn check_batch(&self, batch: &Batch) -> Result<(), &'static str> {
        for (peer, entry) in self.entries.iter().enumerate() {
            if let Some(entry) = entry {
                if &entry.key == batch.key() && !batch.expects(peer) {
                    return Err("Unexpected peer in early message cache");
                }
            }
        }
        Ok(())
    }

    /// Transactional with respect to validation errors: on Err, nothing moves.
    /// Call only after every outbound permit and unacked budget is reserved.
    pub fn fill_batch(&mut self, batch: &mut Batch) -> Result<(), &'static str> {
        self.check_batch(batch)?;
        for (peer, slot) in self.entries.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(|entry| &entry.key == batch.key()) {
                let entry = slot.take().expect("checked early entry");
                self.count -= 1;
                self.bytes -= entry.payload.len();
                // check_batch() established membership; duplicate inserts are safe.
                batch
                    .insert(peer, entry.payload)
                    .expect("validated early peer");
            }
        }
        Ok(())
    }

    pub fn remove_peer(&mut self, peer: PeerIndex) {
        if let Some(entry) = self.entries.get_mut(peer).and_then(Option::take) {
            self.count -= 1;
            self.bytes -= entry.payload.len();
        }
    }
}

/// Bounded record, unchanged: callers must not deliberately reuse old keys.
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

    fn key(s: &'static [u8]) -> Bytes {
        Bytes::from_static(s)
    }
    fn layout() -> Arc<PeerLayout> {
        Arc::new(PeerLayout::new(vec![(2, "B".into()), (7, "C".into())]))
    }

    #[test]
    fn indexed_batch_duplicate_and_unexpected_peer() {
        let mut b = Batch::new(key(b"k"), layout(), Vec::new());
        assert_eq!(b.insert(2, key(b"first")), Ok(true));
        assert_eq!(b.insert(2, key(b"duplicate")), Ok(false));
        assert!(b.insert(1, key(b"bad")).is_err());
        assert!(!b.is_complete());
        assert_eq!(b.insert(7, key(b"second")), Ok(true));
        let (r, slots) = b.take().unwrap();
        assert_eq!(&r["B"][..], b"first");
        assert_eq!(&r["C"][..], b"second");
        assert!(slots.is_empty());
        assert!(slots.capacity() >= 2);
    }

    #[test]
    fn storage_is_reused_without_retaining_old_payloads() {
        let mut b = Batch::new(key(b"one"), layout(), Vec::new());
        b.insert(2, key(b"a")).unwrap();
        b.insert(7, key(b"b")).unwrap();
        let ptr = b.received.as_ptr();
        let (_, recycled) = b.take().unwrap();
        let b = Batch::new(key(b"two"), layout(), recycled);
        assert_eq!(b.received.as_ptr(), ptr);
        assert!(b.received.iter().all(Option::is_none));
        assert!(!b.is_complete());
    }

    #[test]
    fn empty_batch_completes() {
        let b = Batch::new(key(b"k"), Arc::new(PeerLayout::new(Vec::new())), Vec::new());
        assert!(b.is_complete());
        assert!(b.take().unwrap().0.is_empty());
    }

    #[test]
    fn early_validation_does_not_destroy_acknowledged_data() {
        let mut e = EarlyMessages::new(8, 1024);
        e.insert(key(b"k"), 2, key(b"a")).unwrap();
        e.insert(key(b"k"), 7, key(b"b")).unwrap();
        let wrong = Arc::new(PeerLayout::new(vec![(2, "B".into())]));
        let mut b = Batch::new(key(b"k"), wrong, Vec::new());
        assert!(e.fill_batch(&mut b).is_err());
        assert_eq!(e.count, 2);
        assert_eq!(e.bytes, 2);
        let mut b = Batch::new(key(b"k"), layout(), b.recycle());
        e.fill_batch(&mut b).unwrap();
        assert!(b.is_complete());
        assert_eq!((e.count, e.bytes), (0, 0));
    }

    #[test]
    fn early_limits_duplicates_and_unordered_keys() {
        let mut e = EarlyMessages::new(2, 5);
        assert_eq!(e.insert(key(b"z"), 2, key(b"123")), Ok(true));
        assert_eq!(e.insert(key(b"z"), 2, key(b"duplicate")), Ok(false));
        assert!(e.insert(key(b"next"), 2, Bytes::new()).is_err());
        assert!(e.insert(key(b"z"), 7, key(b"456")).is_err());
        e.insert(key(b"a"), 7, key(b"4")).unwrap();
        assert!(e.insert(key(b"c"), 8, Bytes::new()).is_err());
        let mut b = Batch::new(key(b"z"), layout(), Vec::new());
        e.fill_batch(&mut b).unwrap();
        assert_eq!((e.count, e.bytes), (1, 1));
        e.remove_peer(7);
        assert_eq!((e.count, e.bytes), (0, 0));
    }

    #[test]
    fn completed_keys_remain_bounded() {
        let mut h = CompletedKeys::new(2);
        h.insert(key(b"z"));
        h.insert(key(b"a"));
        h.insert(key(b"z"));
        h.insert(key(b"q"));
        assert!(!h.contains(&key(b"z")));
        assert!(h.contains(&key(b"a")) && h.contains(&key(b"q")));
    }
}
