use crate::node::Node;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

pub const ALIGN4: u32 = 3;

#[derive(Error, Debug, PartialEq)]

pub enum ArenaError {
    #[error("allocation failed because arena is full")]
    ArenaFull,
}

pub struct Arena {
    n: AtomicU64,
    pub(crate) buf: Vec<u8>,
}

impl Arena {
    pub fn new(cap: u32) -> Arena {
        Arena {
            n: AtomicU64::new(1),
            buf: vec![0; cap as usize],
        }
    }

    pub fn size(&self) -> u32 {
        let s = self.n.load(Ordering::SeqCst);
        if s > u32::MAX as u64 {
            u32::MAX
        } else {
            s as u32
        }
    }

    pub fn capacity(&self) -> u32 {
        self.buf.len() as u32
    }

    pub fn alloc(&self, size: u32, align: u32, overflow: u32) -> Result<(u32, u32), ArenaError> {
        // Verify that the arena isn't already full.
        let orig_size = self.n.load(Ordering::SeqCst);
        if orig_size > self.buf.len().try_into().unwrap() {
            return Err(ArenaError::ArenaFull);
        }

        // Pad the allocation with enough bytes to ensure the requested alignment.
        let padded = size + align;

        self.n.fetch_add(padded as u64, Ordering::SeqCst);
        let new_size = self.n.load(Ordering::SeqCst);
        if new_size + overflow as u64 > self.buf.len().try_into().unwrap() {
            return Err(ArenaError::ArenaFull);
        }

        // Return the aligned offset.
        let offset = (new_size as u32 - padded + align) & !align;
        Ok((offset, padded))
    }

    pub fn get_bytes_mut(&mut self, offset: u32, size: u32) -> &mut [u8] {
        &mut self.buf[offset as usize..(offset + size) as usize]
    }

    pub fn get_pointer_mut(&mut self, offset: u32) -> *mut Node {
        let pointer = &mut self.buf[offset as usize] as *mut u8;
        pointer.cast()
    }

    pub fn get_pointer_offset(&mut self, node: *const Node) -> u32 {
        if node.is_null() {
            return 0;
        }

        let zero = self.get_pointer_mut(0).as_const();
        (node.addr() - zero.addr()) as u32
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::arena::{Arena, ArenaError, ALIGN4};
    use std::ptr::Unique;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn test_arena_size_overflow() {
        let a = Arena::new(u32::MAX);
        // Allocating under the limit throws no error.
        let res = a.alloc(u16::MAX as u32, 0, 0);
        assert!(res.is_ok());
        let (offset, _) = res.unwrap();
        assert_eq!(1 as u32, offset);
        assert_eq!(u16::MAX as u32 + 1, a.size());

        // Allocating over the limit could cause an accounting
        // overflow if 32-bit arithmetic was used. It shouldn't.
        let res = a.alloc(u32::MAX, 0, 0);
        assert!(res.is_err());
        assert_eq!(ArenaError::ArenaFull, res.unwrap_err());
        assert_eq!(u32::MAX, a.size());

        // Continuing to allocate continues to throw an error.
        let res = a.alloc(u16::MAX as u32, 0, 0);
        assert!(res.is_err());
        assert_eq!(ArenaError::ArenaFull, res.unwrap_err());
        assert_eq!(u32::MAX, a.size());
    }

    #[test]
    fn test_concurrency_alloc() {
        let a = &mut Arena::new(u32::MAX) as *mut Arena;
        let d = Unique::new(a);
        let mut handles = Vec::with_capacity(10);
        let barrier = Arc::new(Barrier::new(10));

        for _ in 0..10 {
            let c = Arc::clone(&barrier);
            handles.push(thread::spawn(move || unsafe {
                println!("before wait");
                c.wait();
                println!("after wait");
                let (e, f) = d.unwrap().as_ref().alloc(u16::MAX as u32, 0, 0).unwrap();
                println!("{},{}", e, f);
                assert_eq!(
                    u16::MAX,
                    d.unwrap().as_mut().get_bytes_mut(e, f).len() as u16
                )
            }));
        }

        // Wait for other threads to finish.
        for handle in handles {
            handle.join().unwrap();
        }

        unsafe {
            assert_eq!(u16::MAX as u32 * 10 + 1, a.as_ref().unwrap().size());
            assert_eq!(u32::MAX, a.as_ref().unwrap().capacity());
        }
    }

    #[test]
    fn test_pointer_offset() {
        let mut a = Arena::new(u32::MAX);
        let (node1_offset, _) = a.alloc(u16::MAX as u32, ALIGN4, 1024).unwrap();
        let node1_p = a.get_pointer_mut(node1_offset);
        let exp_node1_offset = a.get_pointer_offset(node1_p);
        assert_eq!(exp_node1_offset, node1_offset);
        let (node2_offset, _) = a.alloc(u16::MAX as u32, ALIGN4, 1024).unwrap();
        let node2_p = a.get_pointer_mut(node2_offset);
        let exp_node2_offset = a.get_pointer_offset(node2_p);
        assert_eq!(exp_node2_offset, node2_offset)
    }
}
