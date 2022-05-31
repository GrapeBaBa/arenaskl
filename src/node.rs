use crate::arena::{Arena, ArenaError, ALIGN4};
use crate::node::NodeError::{
    ArenaFull, CombinedKeyAndValueTooLarge, InvalidHeight, KeyTooLarge, ValueTooLarge,
};
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

#[allow(dead_code)]
pub const MAX_HEIGHT: usize = 20;
const MAX_NODE_SIZE: usize = mem::size_of::<Node>();
const LINK_SIZE: usize = mem::size_of::<Link>();

#[allow(dead_code)]
#[repr(C)]
pub struct Link {
    prev_offset: AtomicU32,
    next_offset: AtomicU32,
}

#[allow(dead_code)]
impl Link {
    pub fn new(prev_offset: AtomicU32, next_offset: AtomicU32) -> Link {
        Link {
            prev_offset,
            next_offset,
        }
    }
}

#[repr(C)]
pub struct Node {
    key_offset: u32,
    key_size: u32,
    value_size: u32,
    alloc_size: u32,

    tower: [Link; MAX_HEIGHT as usize],
}

#[derive(Error, Debug, PartialEq)]
#[allow(dead_code)]
pub enum NodeError {
    #[error("node allocation failed because arena is full")]
    ArenaFull(ArenaError),

    #[error("the height for node `{0}` is invalid")]
    InvalidHeight(u32),

    #[error("the key size for node `{0}` is invalid")]
    KeyTooLarge(usize),

    #[error("the value size for node `{0}` is invalid")]
    ValueTooLarge(usize),

    #[error("the key size and value size for node `{0}` is invalid")]
    CombinedKeyAndValueTooLarge(usize),
}

#[allow(dead_code)]
impl Node {
    pub fn new_node(
        arena: &mut Arena,
        height: u32,
        key: &[u8],
        value: &[u8],
    ) -> Result<*mut Node, NodeError> {
        if height < 1 || height > MAX_HEIGHT as u32 {
            return Err(InvalidHeight(height));
        }

        let key_size = key.len();
        if key_size > u32::MAX as usize {
            return Err(KeyTooLarge(key_size));
        }

        let value_size = value.len();
        if value_size > u32::MAX as usize {
            return Err(ValueTooLarge(value_size));
        }

        let combined_size = key_size + value_size + MAX_NODE_SIZE;
        if combined_size > u32::MAX as usize {
            return Err(CombinedKeyAndValueTooLarge(combined_size));
        }

        let node = Node::new_raw_node(arena, height, key_size as u32, value_size as u32)?;

        unsafe {
            let key_buf = node.as_mut().unwrap().get_key_mut(arena);
            key_buf.copy_from_slice(key);
            let value_buf = node.as_mut().unwrap().get_value_mut(arena);
            value_buf.copy_from_slice(value);
            Ok(node)
        }
    }

    pub fn new_raw_node(
        arena: &mut Arena,
        height: u32,
        key_size: u32,
        value_size: u32,
    ) -> Result<*mut Node, NodeError> {
        // Compute the amount of the tower that will never be used, since the height
        // is less than maxHeight.
        let unused_size = (MAX_HEIGHT as u32 - height) * LINK_SIZE as u32;
        let node_size = MAX_NODE_SIZE as u32 - unused_size;

        let (node_offset, alloc_size) = arena
            .alloc(node_size + key_size + value_size, ALIGN4, unused_size)
            .map_err(ArenaFull)?;

        let node = unsafe {
            let node = arena.get_pointer_mut(node_offset);
            node.as_mut().unwrap().key_offset = node_offset + node_size;
            node.as_mut().unwrap().key_size = key_size;
            node.as_mut().unwrap().value_size = value_size;
            node.as_mut().unwrap().alloc_size = alloc_size;
            node
        };

        Ok(node)
    }

    pub fn get_key_mut<'a>(&'a self, arena: &'a mut Arena) -> &mut [u8] {
        arena.get_bytes_mut(self.key_offset, self.key_size)
    }

    pub fn get_value_mut<'a>(&'a self, arena: &'a mut Arena) -> &mut [u8] {
        arena.get_bytes_mut(self.key_offset + self.key_size, self.value_size)
    }

    pub fn next_offset(&self, h: usize) -> u32 {
        self.tower[h].next_offset.load(Ordering::SeqCst)
    }

    pub fn prev_offset(&self, h: usize) -> u32 {
        self.tower[h].prev_offset.load(Ordering::SeqCst)
    }

    pub fn cas_next_offset(&self, h: usize, current: u32, new: u32) -> Result<u32, u32> {
        self.tower[h]
            .next_offset
            .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
    }

    pub fn cas_prev_offset(&self, h: usize, current: u32, new: u32) -> Result<u32, u32> {
        self.tower[h]
            .prev_offset
            .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::node::Node;

    #[test]
    fn test_new_raw_node() {
        let mut a = Arena::new(u32::MAX);

        unsafe {
            let raw_node = Node::new_raw_node(&mut a, 12, 1024, 2048).unwrap();
            let key_offset = raw_node.as_mut().unwrap().key_offset;
            let key_size = raw_node.as_mut().unwrap().key_size;
            let value_size = raw_node.as_mut().unwrap().value_size;
            let alloc_size = raw_node.as_mut().unwrap().alloc_size;
            let b = a.get_bytes_mut(4, alloc_size);
            assert_eq!(key_offset, u32::from_le_bytes(b[0..4].try_into().unwrap()));
            assert_eq!(key_size, u32::from_le_bytes(b[4..8].try_into().unwrap()));
            assert_eq!(value_size, u32::from_le_bytes(b[8..12].try_into().unwrap()));
            assert_eq!(
                alloc_size,
                u32::from_le_bytes(b[12..16].try_into().unwrap())
            );
            assert_eq!(alloc_size, b.len() as u32)
        }
    }

    #[test]
    fn test_new_node() {
        let mut a = Arena::new(u32::MAX);

        unsafe {
            let node = Node::new_node(
                &mut a,
                12,
                &[1u8, 1u8, 1u8, 1u8],
                &[1u8, 1u8, 1u8, 1u8, 1u8],
            )
                .unwrap();
            let key_offset = node.as_mut().unwrap().key_offset;
            let key_size = node.as_mut().unwrap().key_size;
            let value_size = node.as_mut().unwrap().value_size;
            let alloc_size = node.as_mut().unwrap().alloc_size;
            let b = a.get_bytes_mut(4, alloc_size);
            assert_eq!(key_offset, u32::from_le_bytes(b[0..4].try_into().unwrap()));
            assert_eq!(key_size, u32::from_le_bytes(b[4..8].try_into().unwrap()));
            assert_eq!(value_size, u32::from_le_bytes(b[8..12].try_into().unwrap()));
            assert_eq!(
                alloc_size,
                u32::from_le_bytes(b[12..16].try_into().unwrap())
            );
            assert_eq!(alloc_size, b.len() as u32)
        }
    }
}
