use std::lazy::SyncOnceCell;
use crate::arena::Arena;
use crate::node::Node;
use std::ptr::Unique;
use std::sync::atomic::AtomicU32;
use crate::node;

const P_VALUE: f64 = 1.0f64 / std::f64::consts::E;
static PROBABILITIES: SyncOnceCell<[u32; node::MAX_HEIGHT]> = SyncOnceCell::new();

pub fn get_probabilities() -> &'static [u32; node::MAX_HEIGHT] {
    PROBABILITIES.get_or_init(|| {
        let mut probabilities = [0u32; node::MAX_HEIGHT];
        let mut p = 1.0f64;
        for i in 0..node::MAX_HEIGHT {
            probabilities[i] = (u32::MAX as f64 * p) as u32;
            p *= P_VALUE
        }
        probabilities
    })
}

pub struct Skiplist {
    arena: Option<Unique<Arena>>,
    head: Option<Unique<Node>>,
    tail: Option<Unique<Node>>,
    height: AtomicU32, // Current height. 1 <= height <= maxHeight. CAS.

    // If set to true by tests, then extra delays are added to make it easier to
    // detect unusual race conditions.
    testing: bool,
}

impl Skiplist {
    pub fn get_next_mut(&mut self, node: Unique<Node>, h: usize) -> Option<Unique<Node>> {
        unsafe {
            let offset = node.as_ref().next_offset(h);
            let next = self.arena.unwrap().as_mut().get_pointer_mut(offset);
            Unique::new(next)
        }
    }

    pub fn get_prev_mut(&mut self, node: Unique<Node>, h: usize) -> Option<Unique<Node>> {
        unsafe {
            let offset = node.as_ref().prev_offset(h);
            let prev = self.arena.unwrap().as_mut().get_pointer_mut(offset);
            Unique::new(prev)
        }
    }
}

pub struct Splice {
    prev: Unique<*mut Node>,
    next: Unique<*mut Node>,
}

impl Splice {
    pub fn new(prev: Unique<*mut Node>, next: Unique<*mut Node>) -> Splice {
        Splice { prev, next }
    }
}

pub struct Inserter {
    spl: [Option<Splice>; node::MAX_HEIGHT],
    height: u32,
}
