use crate::arena::Arena;
use crate::iterator::{Iter, ITER_POOL};
use crate::node;
use crate::node::{Node, NodeError, MAX_HEIGHT};
use lifeguard::Recycled;
use rand::RngCore;
use std::lazy::SyncOnceCell;
use std::ptr::Unique;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use thiserror::Error;

const P_VALUE: f64 = 1.0f64 / std::f64::consts::E;
const INTERNAL_TRAILER_LEN: usize = 8;
static PROBABILITIES: SyncOnceCell<[u32; node::MAX_HEIGHT]> = SyncOnceCell::new();

pub fn get_probabilities() -> &'static [u32; node::MAX_HEIGHT] {
    PROBABILITIES.get_or_init(|| {
        let mut probabilities = [0u32; node::MAX_HEIGHT];
        let mut p = 1.0f64;

        probabilities.iter_mut().for_each(|probability| {
            *probability = (u32::MAX as f64 * p) as u32;
            p *= P_VALUE
        });
        probabilities
    })
}

type InternalKeyKind = u8;

pub const INTERNAL_KEY_KIND_DELETE: InternalKeyKind = 0;
pub const INTERNAL_KEY_KIND_SET: InternalKeyKind = 1;
pub const INTERNAL_KEY_KIND_MAX: InternalKeyKind = 21;
pub const INTERNAL_KEY_KIND_INVALID: InternalKeyKind = 255;

pub const INTERNAL_KEY_SEQ_NUM_MAX: u64 = (1 << 56) - 1;

#[derive(Error, Debug, PartialEq)]
pub enum SKLError {
    #[error(transparent)]
    NodeError(#[from] NodeError),

    #[error("the key already exist")]
    RecordExist,
}

#[derive(Clone)]
pub struct InternalKey {
    pub user_key: Vec<u8>,
    pub trailer: u64,
}

impl InternalKey {
    pub fn new(user_key: &[u8], seq_num: u64, kind: InternalKeyKind) -> InternalKey {
        InternalKey {
            user_key: user_key.to_vec(),
            trailer: (seq_num << 8) | (kind as u64),
        }
    }

    pub fn new_trailer(seq_num: u64, kind: InternalKeyKind) -> u64 {
        (seq_num << 8) | (kind as u64)
    }

    pub fn size(&self) -> usize {
        self.user_key.len() + 8
    }

    pub fn encode(&self, buf: &mut [u8]) {
        buf[..self.user_key.len()].copy_from_slice(self.user_key.as_ref());
        buf[self.user_key.len()..].copy_from_slice(self.trailer.to_ne_bytes().as_ref());
    }

    pub fn decode(encoded_key: &[u8]) -> InternalKey {
        let n = encoded_key.len() - INTERNAL_TRAILER_LEN;
        let (user_key, trailer) = if encoded_key.len() >= 8 {
            (
                Vec::from(&encoded_key[..n]),
                u64::from_ne_bytes(encoded_key[n..].try_into().unwrap()),
            )
        } else {
            (vec![], INTERNAL_KEY_KIND_INVALID as u64)
        };
        InternalKey { user_key, trailer }
    }
}

pub struct Skiplist {
    pub(crate) arena: Unique<Arena>,
    pub(crate) head: Unique<Node>,
    pub(crate) tail: Unique<Node>,
    pub(crate) height: AtomicU32, // Current height. 1 <= height <= maxHeight. CAS.

    // If set to true by tests, then extra delays are added to make it easier to
    // detect unusual race conditions.
    pub(crate) testing: bool,
}

impl Skiplist {
    pub fn new(arena: &mut Arena) -> Result<Skiplist, SKLError> {
        let head = Node::new_raw_node(arena, MAX_HEIGHT as u32, 0u32, 0u32)
            .map_err(SKLError::NodeError)?;
        unsafe {
            head.as_mut().unwrap().key_offset = 0u32;
        }

        let tail = Node::new_raw_node(arena, MAX_HEIGHT as u32, 0u32, 0u32)
            .map_err(SKLError::NodeError)?;
        unsafe {
            tail.as_mut().unwrap().key_offset = 0u32;
        }

        let head_offset = arena.get_pointer_offset(head.as_const());
        let tail_offset = arena.get_pointer_offset(tail.as_const());
        unsafe {
            for i in 0..MAX_HEIGHT {
                head.as_mut().unwrap().tower[i].next_offset = AtomicU32::from(tail_offset);
                tail.as_mut().unwrap().tower[i].prev_offset = AtomicU32::from(head_offset);
            }
        }

        let skl = Skiplist {
            arena: Unique::from(arena),
            head: Unique::new(head).unwrap(),
            tail: Unique::new(tail).unwrap(),
            height: AtomicU32::new(1),
            testing: false,
        };

        Ok(skl)
    }

    pub fn get_next_mut(&mut self, node: Unique<Node>, h: usize) -> Unique<Node> {
        unsafe {
            let offset = node.as_ref().next_offset(h);
            let next = self.arena.as_mut().get_pointer_mut(offset);
            Unique::new(next).unwrap()
        }
    }

    pub fn get_prev_mut(&mut self, node: Unique<Node>, h: usize) -> Unique<Node> {
        unsafe {
            let offset = node.as_ref().prev_offset(h);
            let prev = self.arena.as_mut().get_pointer_mut(offset);
            Unique::new(prev).unwrap()
        }
    }

    pub fn height(&self) -> u32 {
        self.height.load(Ordering::SeqCst)
    }

    pub fn arena(&self) -> Unique<Arena> {
        self.arena
    }

    pub fn size(&self) -> u32 {
        unsafe { self.arena.as_ref().size() }
    }

    pub fn reset(&mut self, arena: &mut Arena) -> Result<(), SKLError> {
        let skl = Skiplist::new(arena)?;
        *self = skl;
        Ok(())
    }

    pub fn iter<'a>(&mut self, lower: &'a [u8], upper: &'a [u8]) -> Recycled<'a, Iter> {
        let mut iter = unsafe { ITER_POOL.with(|p| p.as_ptr().as_ref().unwrap().new()) };
        iter.upper = Vec::from(upper);
        iter.lower = Vec::from(lower);
        iter.list = Unique::new(self as *mut Skiplist).unwrap();
        iter.node = self.head;
        iter
    }

    pub fn add(&mut self, key: &InternalKey, value: &[u8]) -> Result<(), SKLError> {
        let mut ins = Inserter::new();
        self.add_internal(key, value, &mut ins)
    }

    pub fn find_splice(&mut self, key: &InternalKey, ins: &mut Inserter) -> bool {
        let list_height = self.height();
        let mut level: u32 = 0u32;

        let mut prev = self.head;
        let mut found: bool = false;
        if ins.height < list_height {
            // Our cached height is less than the list height, which means there were
            // inserts that increased the height of the list. Recompute the splice from
            // scratch.
            ins.height = list_height;
            level = ins.height;
        } else {
            while level < list_height {
                let spl = &ins.spl[level as usize];
                if self
                    .get_next_mut(spl.prev.unwrap(), level as usize)
                    .as_ptr()
                    != spl.next.unwrap().as_ptr()
                {
                    // One or more nodes have been inserted between the splice at this
                    // level.
                    level += 1;
                    continue;
                }
                if spl.prev.unwrap().as_ptr() != self.head.as_ptr()
                    && !self.key_is_after_node(spl.prev.unwrap(), key)
                {
                    // Key lies before splice.
                    level = list_height;
                    break;
                }
                if spl.next.unwrap().as_ptr() != self.tail.as_ptr()
                    && self.key_is_after_node(spl.next.unwrap(), key)
                {
                    // Key lies after splice.
                    level = list_height;
                    break;
                }
                // The splice brackets the key!
                prev = spl.prev.unwrap();
                break;
            }
        }

        for l in (0..=(level as i32 - 1)).rev() {
            let next: Unique<Node>;
            (prev, next, found) = self.find_splice_for_level(key, l as usize, prev);
            ins.spl[l as usize].init(prev, next);
        }

        found
    }

    pub(crate) fn find_splice_for_level(
        &mut self,
        key: &InternalKey,
        level: usize,
        start: Unique<Node>,
    ) -> (Unique<Node>, Unique<Node>, bool) {
        let mut prev = start;
        let mut found = false;
        let mut next: Unique<Node>;
        unsafe {
            loop {
                // Assume prev.key < key.
                next = self.get_next_mut(prev, level);
                if next.as_ptr() == self.tail.as_ptr() {
                    // Tail node, so done.
                    break;
                }
                let next_ref = next.as_ref();
                let next_key = self
                    .arena
                    .as_mut()
                    .get_bytes_mut(next_ref.key_offset, next_ref.key_size);
                let n = next_ref.key_size as usize - 8;
                let cmp = key.user_key.as_slice().cmp(&next_key[..n]);
                if cmp.is_lt() {
                    break;
                }
                if cmp.is_eq() {
                    // User-key equality.
                    let next_trailer = if next_ref.key_size >= 8 {
                        u64::from_ne_bytes(next_key[n..].try_into().unwrap())
                    } else {
                        INTERNAL_KEY_KIND_INVALID as u64
                    };

                    if key.trailer == next_trailer {
                        // Internal key equality.
                        found = true;
                        break;
                    }
                    if key.trailer > next_trailer {
                        // We are done for this level, since prev.key < key < next.key.
                        break;
                    }
                }

                // Keep moving right on this level.
                prev = next
            }
        }

        (prev, next, found)
    }

    fn new_node(
        &mut self,
        key: &InternalKey,
        value: &[u8],
    ) -> Result<(Unique<Node>, u32), SKLError> {
        let height = Skiplist::random_height();
        let node = Node::new_node(unsafe { self.arena.as_mut() }, height, key, value)
            .map_err(SKLError::NodeError)?;
        let mut list_height = self.height();
        while height > list_height {
            let res = self.height.compare_exchange(
                list_height,
                height,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if res.is_ok() {
                break;
            }
            list_height = self.height()
        }

        Ok((Unique::new(node).unwrap(), height))
    }

    fn add_internal(
        &mut self,
        key: &InternalKey,
        value: &[u8],
        ins: &mut Inserter,
    ) -> Result<(), SKLError> {
        if self.find_splice(key, ins) {
            // Found a matching node, but handle case where it's been deleted.
            return Err(SKLError::RecordExist);
        }

        if self.testing {
            // Add delay to make it easier to test race between this thread
            // and another thread that sees the intermediate state between
            // finding the splice and using it.
            thread::yield_now();
        }

        let (node, height) = self.new_node(key, value)?;
        let node_offset =
            unsafe { self.arena.as_mut() }.get_pointer_offset(node.as_ptr().as_const());

        // We always insert from the base level and up. After you add a node in base
        // level, we cannot create a node in the level above because it would have
        // discovered the node in the base level.
        let mut found;
        let mut invalidate_splice = false;
        for i in 0..height as usize {
            let mut prev = ins.spl[i].prev;
            let mut next = ins.spl[i].next;

            if prev.is_none() {
                // New node increased the height of the skiplist, so assume that the
                // new level has not yet been populated.
                if next.is_some() {
                    panic!("next is expected to be nil, since prev is nil")
                }

                prev = Some(self.head);
                next = Some(self.tail);
            }

            // +----------------+     +------------+     +----------------+
            // |      prev      |     |     nd     |     |      next      |
            // | prevNextOffset |---->|            |     |                |
            // |                |<----| prevOffset |     |                |
            // |                |     | nextOffset |---->|                |
            // |                |     |            |<----| nextPrevOffset |
            // +----------------+     +------------+     +----------------+
            //
            // 1. Initialize prevOffset and nextOffset to point to prev and next.
            // 2. CAS prevNextOffset to repoint from next to nd.
            // 3. CAS nextPrevOffset to repoint from prev to nd.
            unsafe {
                loop {
                    let prev_offset = self
                        .arena
                        .as_mut()
                        .get_pointer_offset(prev.unwrap().as_ptr());
                    let next_offset = self
                        .arena
                        .as_mut()
                        .get_pointer_offset(next.unwrap().as_ptr());
                    node.as_ptr().as_mut().unwrap().tower[i].init(prev_offset, next_offset);

                    // Check whether next has an updated link to prev. If it does not,
                    // that can mean one of two things:
                    //   1. The thread that added the next node hasn't yet had a chance
                    //      to add the prev link (but will shortly).
                    //   2. Another thread has added a new node between prev and next.
                    let next_prev_offset = next.unwrap().as_ref().prev_offset(i);
                    if next_prev_offset != prev_offset {
                        // Determine whether #1 or #2 is true by checking whether prev
                        // is still pointing to next. As long as the atomic operations
                        // have at least acquire/release semantics (no need for
                        // sequential consistency), this works, as it is equivalent to
                        // the "publication safety" pattern.
                        let prev_next_offset = prev.unwrap().as_ref().next_offset(i);
                        if prev_next_offset == next_offset {
                            // Ok, case #1 is true, so help the other thread along by
                            // updating the next node's prev link.
                            next.unwrap().as_mut().cas_prev_offset(
                                i,
                                next_prev_offset,
                                prev_offset,
                            );
                        }
                    }

                    if prev
                        .unwrap()
                        .as_mut()
                        .cas_next_offset(i, next_offset, node_offset)
                    {
                        // Managed to insert nd between prev and next, so update the next
                        // node's prev link and go to the next level.
                        if self.testing {
                            // Add delay to make it easier to test race between this thread
                            // and another thread that sees the intermediate state between
                            // setting next and setting prev.
                            thread::yield_now()
                        }

                        next.unwrap()
                            .as_mut()
                            .cas_prev_offset(i, prev_offset, node_offset);
                        break;
                    }

                    // CAS failed. We need to recompute prev and next. It is unlikely to
                    // be helpful to try to use a different level as we redo the search,
                    // because it is unlikely that lots of nodes are inserted between prev
                    // and next.
                    let (prev_opt, next_opt, is_exist) =
                        self.find_splice_for_level(key, i, prev.unwrap());
                    prev = Some(prev_opt);
                    next = Some(next_opt);
                    found = is_exist;
                    if found {
                        if i != 0 {
                            panic!(
                                "how can another thread have inserted a node at a non-base level?"
                            )
                        }

                        return Err(SKLError::RecordExist);
                    }
                    invalidate_splice = true
                }
            }
        }

        // If we had to recompute the splice for a level, invalidate the entire
        // cached splice.
        if invalidate_splice {
            ins.height = 0
        } else {
            // The splice was valid. We inserted a node between spl[i].prev and
            // spl[i].next. Optimistically update spl[i].prev for use in a subsequent
            // call to add.
            for i in 0..height as usize {
                ins.spl[i].prev = Some(node)
            }
        }

        Ok(())
    }

    fn key_is_after_node(&mut self, node: Unique<Node>, key: &InternalKey) -> bool {
        let node_ref = unsafe { node.as_ref() };
        let node_key =
            unsafe { self.arena.as_mut() }.get_bytes_mut(node_ref.key_offset, node_ref.key_size);
        let n = node_ref.key_size as usize - 8;
        let cmp = &node_key[..n].cmp(&key.user_key);
        if cmp.is_lt() {
            return true;
        }
        if cmp.is_gt() {
            return false;
        }

        let node_trailer = if node_ref.key_size >= 8 {
            u64::from_ne_bytes(node_key[n..].try_into().unwrap())
        } else {
            INTERNAL_KEY_KIND_INVALID as u64
        };

        if node_trailer == key.trailer {
            return false;
        }

        key.trailer < node_trailer
    }

    fn random_height() -> u32 {
        let rnd = rand::thread_rng().next_u32();

        let mut h = 1u32;
        let a = get_probabilities();
        let p = a[h as usize];
        while h < node::MAX_HEIGHT as u32 && rnd <= p {
            h += 1;
        }
        h
    }
}

#[derive(Clone, Debug)]
pub struct Splice {
    pub(crate) prev: Option<Unique<Node>>,
    pub(crate) next: Option<Unique<Node>>,
}

impl Splice {
    pub fn empty() -> Splice {
        Splice {
            prev: None,
            next: None,
        }
    }

    pub fn new(prev: Unique<Node>, next: Unique<Node>) -> Splice {
        Splice {
            prev: Some(prev),
            next: Some(next),
        }
    }

    pub fn init(&mut self, prev: Unique<Node>, next: Unique<Node>) {
        self.prev = Some(prev);
        self.next = Some(next);
    }
}

pub struct Inserter {
    pub(crate) spl: [Splice; node::MAX_HEIGHT],
    pub(crate) height: u32,
}

impl Inserter {
    pub fn new() -> Inserter {
        let spl = vec![Splice::empty(); MAX_HEIGHT];
        Inserter {
            spl: spl.try_into().unwrap(),
            height: 0,
        }
    }
}

impl Default for Inserter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::iterator::{Iter, ITER_POOL};
    use crate::node::MAX_HEIGHT;
    use crate::skl::{
        Inserter, InternalKey, SKLError, Skiplist, INTERNAL_KEY_KIND_DELETE,
        INTERNAL_KEY_KIND_INVALID, INTERNAL_KEY_KIND_SET,
    };
    use std::sync::atomic::Ordering;

    #[test]
    fn test_make_trailer() {
        let t = InternalKey::new_trailer(2u64, INTERNAL_KEY_KIND_SET);
        assert_eq!(2u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64, t);
        let t1 = InternalKey::new_trailer(3u64, INTERNAL_KEY_KIND_INVALID);
        assert_eq!(3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_INVALID as u64, t1);
    }

    #[test]
    fn test_make_internal_key() {
        let user_key = [1u8, 2u8];
        let key = InternalKey::new(&user_key, 3u64, INTERNAL_KEY_KIND_SET);
        assert_eq!(key.user_key.len(), 2);
        assert_eq!(key.user_key[0], 1u8);
        assert_eq!(key.user_key[1], 2u8);
        assert_eq!(
            key.trailer,
            3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64
        );
    }

    #[test]
    fn test_random_height() {
        let h = Skiplist::random_height();
        assert!(h >= 1 && h <= 20);
    }

    #[test]
    fn test_new_skiplist() {
        let mut a = Arena::new(u32::MAX);
        let b = &mut a as *mut Arena;

        let skl_res = Skiplist::new(&mut a);
        assert!(skl_res.is_ok());
        let skl = skl_res.unwrap();
        assert_eq!(skl.arena.as_ptr(), b);
        assert_eq!(skl.height.load(Ordering::Acquire), 1u32);
        unsafe { assert_eq!(skl.head.as_ref().key_offset, 0u32) }
        unsafe { assert_eq!(skl.head.as_ref().key_size, 0u32) }
        unsafe { assert_eq!(skl.tail.as_ref().key_offset, 0u32) }
        unsafe { assert_eq!(skl.tail.as_ref().key_size, 0u32) }

        let tail_p = skl.tail.as_ptr();
        let tail_offset = a.get_pointer_offset(tail_p);
        unsafe {
            assert_eq!(
                skl.head.as_ref().tower[MAX_HEIGHT as usize - 1]
                    .next_offset
                    .load(Ordering::Acquire),
                tail_offset
            )
        }
        let head_p = skl.head.as_ptr();
        let head_offset = a.get_pointer_offset(head_p);
        unsafe {
            assert_eq!(
                skl.tail.as_ref().tower[MAX_HEIGHT as usize - 1]
                    .prev_offset
                    .load(Ordering::Acquire),
                head_offset
            )
        }
    }

    #[test]
    fn test_reset_skiplist() {
        let mut a = Arena::new(u32::MAX);
        let b = &mut a as *mut Arena;

        let skl_res = Skiplist::new(&mut a);
        assert!(skl_res.is_ok());
        let mut skl = skl_res.unwrap();
        assert_eq!(skl.arena.as_ptr(), b);

        let mut c = Arena::new(u32::MAX);
        let d = &mut c as *mut Arena;
        let _ = skl.reset(&mut c);
        assert_eq!(skl.arena.as_ptr(), d);
    }

    #[test]
    fn test_basic_add() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let key1 = InternalKey::new(&[1u8], 1u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key1, &[1u8]);
        let res = skl.add(&key1, &[2u8]);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let key2 = InternalKey::new(&[2u8], 1u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key2, &[1u8]);
        let key3 = InternalKey::new(&[1u8], 2u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key3, &[1u8]);
        let res = skl.add(&key2, &[2u8]);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add(&key3, &[2u8]);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());

        let key4 = InternalKey::new(&[3u8], 1u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key4, &[1u8]);
        let key5 = InternalKey::new(&[4u8], 2u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key5, &[1u8]);
        let key6 = InternalKey::new(&[5u8], 2u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key6, &[1u8]);

        let res = skl.add(&key6, &[2u8]);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
    }

    #[test]
    fn test_inserter_add() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let mut ins = Inserter::default();
        let key3 = InternalKey::new(&[1u8], 2u64, INTERNAL_KEY_KIND_SET);
        let key1 = InternalKey::new(&[1u8], 1u64, INTERNAL_KEY_KIND_SET);
        let key2 = InternalKey::new(&[2u8], 1u64, INTERNAL_KEY_KIND_SET);
        let key4 = InternalKey::new(&[3u8], 1u64, INTERNAL_KEY_KIND_SET);
        let key5 = InternalKey::new(&[4u8], 2u64, INTERNAL_KEY_KIND_SET);
        let key6 = InternalKey::new(&[5u8], 2u64, INTERNAL_KEY_KIND_SET);
        let key7 = InternalKey::new(&[5u8], 2u64, INTERNAL_KEY_KIND_DELETE);
        let _ = skl.add_internal(&key3, &[1u8], &mut ins);
        let _ = skl.add_internal(&key1, &[1u8], &mut ins);
        let _ = skl.add_internal(&key2, &[1u8], &mut ins);
        let _ = skl.add_internal(&key4, &[1u8], &mut ins);
        let _ = skl.add_internal(&key5, &[1u8], &mut ins);
        let _ = skl.add_internal(&key6, &[1u8], &mut ins);
        let _ = skl.add_internal(&key7, &[1u8], &mut ins);
        let res = skl.add_internal(&key7, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key6, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key5, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key4, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key3, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key2, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let res = skl.add_internal(&key1, &[1u8], &mut ins);
        assert!(res.is_err());
        assert_eq!(SKLError::RecordExist, res.unwrap_err());
        let key8 = InternalKey::new(&[5u8], 2u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add_internal(&key8, &[1u8], &mut ins);
    }

    #[test]
    fn test_new_iter() {
        let mut a = Arena::new(u32::MAX);

        let skl_res = Skiplist::new(&mut a);
        assert!(skl_res.is_ok());
        let mut skl = skl_res.unwrap();
        let iter = skl.iter(&[] as &[u8], &[] as &[u8]);
        ITER_POOL.with_borrow(|p| assert_eq!(127, p.size()));
        Iter::close(iter);
        ITER_POOL.with_borrow(|p| assert_eq!(128, p.size()));

        assert!(!skl.head.as_ptr().is_null())
    }
}
