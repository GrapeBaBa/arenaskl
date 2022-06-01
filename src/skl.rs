use crate::arena::Arena;
use crate::node;
use crate::node::Node;
use std::lazy::SyncOnceCell;
use std::ptr::Unique;
use std::sync::atomic::AtomicU32;
use thiserror::Error;

const P_VALUE: f64 = 1.0f64 / std::f64::consts::E;
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
pub enum SKLError {}

pub struct InternalKey {
    pub user_key: Vec<u8>,
    pub trailer: u64,
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

    pub fn find_splice_for_level(&mut self, key: InternalKey, level: usize, start: Unique<Node>) -> (Unique<Node>, Unique<Node>, bool) {
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
                        u64::from_le_bytes(next_key[n..].try_into().unwrap())
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

    pub fn key_is_after_node(&mut self, node: Unique<Node>, key: InternalKey) -> bool {
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
            u64::from_le_bytes(node_key[n..].try_into().unwrap())
        } else {
            INTERNAL_KEY_KIND_INVALID as u64
        };

        if node_trailer == key.trailer {
            return false;
        }

        key.trailer < node_trailer
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
    spl: [Option<Splice>; node::MAX_HEIGHT],
    height: u32,
}

pub fn make_internal_key(user_key: &[u8], seq_num: u64, kind: InternalKeyKind) -> InternalKey {
    InternalKey {
        user_key: user_key.to_vec(),
        trailer: (seq_num << 8) | (kind as u64),
    }
}

pub fn make_trailer(seq_num: u64, kind: InternalKeyKind) -> u64 {
    (seq_num << 8) | (kind as u64)
}
