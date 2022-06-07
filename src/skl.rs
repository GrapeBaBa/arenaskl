use std::fmt::Error;
use crate::arena::Arena;
use crate::node;
use crate::node::{MAX_HEIGHT, Node, NodeError};
use std::lazy::SyncOnceCell;
use std::ptr::Unique;
use std::sync::atomic::{AtomicU32, Ordering};
use rand::RngCore;
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
pub const INTERNAL_KEY_KIND_INVALID: InternalKeyKind = 255;

#[derive(Error, Debug, PartialEq)]
pub enum SKLError {
    #[error(transparent)]
    NodeError(#[from] NodeError)
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
        let (user_key, trailer) =
            if encoded_key.len() >= 8 {
                (Vec::from(&encoded_key[..n]), u64::from_ne_bytes(encoded_key[n..].try_into().unwrap()))
            } else {
                (vec![], INTERNAL_KEY_KIND_INVALID as u64)
            };
        InternalKey {
            user_key,
            trailer,
        }
    }
}

pub struct Skiplist {
    arena: Unique<Arena>,
    head: Unique<Node>,
    tail: Unique<Node>,
    height: AtomicU32, // Current height. 1 <= height <= maxHeight. CAS.

    // If set to true by tests, then extra delays are added to make it easier to
    // detect unusual race conditions.
    testing: bool,
}

impl Skiplist {
    pub fn new(arena: &mut Arena) -> Result<Skiplist, SKLError> {
        let head = Node::new_raw_node(arena, MAX_HEIGHT as u32, 0u32, 0u32).map_err(|e| SKLError::NodeError(e))?;
        unsafe { head.as_mut().unwrap().key_offset = 0u32; }

        let tail = Node::new_raw_node(arena, MAX_HEIGHT as u32, 0u32, 0u32).map_err(|e| SKLError::NodeError(e))?;
        unsafe { tail.as_mut().unwrap().key_offset = 0u32; }

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

    pub fn height(&self) -> u32 { self.height.load(Ordering::SeqCst) }

    pub fn arena(&self) -> Unique<Arena> { self.arena }

    pub fn size(&self) -> u32 { unsafe { self.arena.as_ref().size() } }

    pub fn reset(&mut self, arena: &mut Arena) -> Result<(), SKLError> {
        let skl = Skiplist::new(arena)?;

        unsafe {
            *self = skl
        }
        Ok(())
    }

    pub fn find_splice(&mut self, key: &InternalKey, ins: &mut Inserter) -> bool {
        let list_height = self.height();
        let mut level: u32 = 0u32;

        let mut prev = self.head;
        let mut is_exist: bool = false;
        if ins.height < list_height {
            // Our cached height is less than the list height, which means there were
            // inserts that increased the height of the list. Recompute the splice from
            // scratch.
            ins.height = list_height;
            level = ins.height;
        } else {
            while level < list_height {
                let spl = &ins.spl[level as usize];
                if self.get_next_mut(spl.prev, level as usize).as_ptr() != spl.next.as_ptr() {
                    // One or more nodes have been inserted between the splice at this
                    // level.
                    level += 1;
                    continue;
                }
                if spl.prev.as_ptr() != self.head.as_ptr() && !self.key_is_after_node(spl.prev, key) {
                    // Key lies before splice.
                    level = list_height;
                    break;
                }
                if spl.next.as_ptr() != self.tail.as_ptr() && self.key_is_after_node(spl.next, key) {
                    // Key lies after splice.
                    level = list_height;
                    break;
                }
                // The splice brackets the key!
                prev = spl.prev;
                break;
            }
        }

        for l in (0..=level - 1).rev() {
            let (prev, mut next, found) = self.find_splice_for_level(key, l as usize, prev);
            is_exist = found;
            if next.as_ptr().is_null() {
                next = self.tail;
            }
            ins.spl[l as usize] = Splice::new(prev, next);
        }

        is_exist
    }

    fn find_splice_for_level(&mut self, key: &InternalKey, level: usize, start: Unique<Node>) -> (Unique<Node>, Unique<Node>, bool) {
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
                let next_key = self.arena.as_mut().get_bytes_mut(next_ref.key_offset, next_ref.key_size);
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

    fn new_node(&mut self, key: &InternalKey, value: &[u8]) -> Result<(Unique<Node>, u32), SKLError> {
        let height = Skiplist::random_height();
        let node = Node::new_node(unsafe { self.arena.as_mut() }, height, key, value).map_err(|e| SKLError::NodeError(e))?;
        let mut list_height = self.height();
        while height > list_height {
            let res = self.height.compare_exchange(list_height, height, Ordering::SeqCst, Ordering::SeqCst);
            if res.is_ok() {
                break;
            }
            list_height = self.height()
        }

        Ok((Unique::new(node).unwrap(), height))
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

pub struct Splice {
    prev: Unique<Node>,
    next: Unique<Node>,
}

impl Splice {
    pub fn new(prev: Unique<Node>, next: Unique<Node>) -> Splice {
        Splice { prev, next }
    }
}

pub struct Inserter {
    spl: [Splice; node::MAX_HEIGHT],
    height: u32,
}


#[cfg(test)]
mod tests {
    use std::mem::transmute;
    use std::sync::atomic::Ordering;
    use crate::arena::Arena;
    use crate::node::{MAX_HEIGHT, Node};
    use crate::skl::{INTERNAL_KEY_KIND_INVALID, INTERNAL_KEY_KIND_SET, InternalKey, Skiplist};

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
        assert_eq!(key.trailer, 3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64);
    }

    #[test]
    fn test_random_height() {
        let h = Skiplist::random_height();
        assert!(h >= 1 && h <= 20)
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
        unsafe { assert_eq!(skl.head.as_ref().tower[MAX_HEIGHT as usize - 1].next_offset.load(Ordering::Acquire), tail_offset) }
        let head_p = skl.head.as_ptr();
        let head_offset = a.get_pointer_offset(head_p);
        unsafe { assert_eq!(skl.tail.as_ref().tower[MAX_HEIGHT as usize - 1].prev_offset.load(Ordering::Acquire), head_offset) }
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
}