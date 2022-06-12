use crate::node::Node;
use crate::skl::{
    InternalKey, SKLError, Skiplist, INTERNAL_KEY_KIND_MAX, INTERNAL_KEY_SEQ_NUM_MAX,
};
use lifeguard::{InitializeWith, Pool, Recycleable, Recycled};
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::ptr::Unique;
use std::sync::Arc;
use std::thread::LocalKey;

// thread_local!(pub static ITER_POOL:RefCell<Pool<InternalIter >> = panic!("!"));
thread_local!(pub static ITER_POOL:RefCell<Pool<Iter >> = RefCell::new(Pool::with_size_and_max(128, 2048)));

pub struct Iter {
    pub(crate) list: *const Skiplist,
    pub(crate) node: *const Node,
    pub(crate) key: InternalKey,
    pub(crate) lower: Vec<u8>,
    pub(crate) upper: Vec<u8>,
}

impl Iter {
    // Close resets the iterator.
    pub fn close(iter: Recycled<Iter>) {
        drop(iter)
    }

    pub fn seek_for_base_splice(&self, key: &[u8]) -> (Unique<Node>, Unique<Node>, bool) {
        let i_key = InternalKey::new(key, INTERNAL_KEY_SEQ_NUM_MAX, INTERNAL_KEY_KIND_MAX);
        let mut level = (unsafe { self.list.as_ref() }.unwrap().height() - 1) as usize;

        let mut prev = unsafe { self.list.as_ref() }.unwrap().head;
        let mut next: Unique<Node> = Unique::dangling();
        let mut found: bool = false;
        loop {
            unsafe {
                (prev, next, found) = self
                    .list
                    .as_mut()
                    .as_mut()
                    .unwrap()
                    .find_splice_for_level(&i_key, level, prev);
            }

            if found {
                if level != 0 {
                    // next is pointing at the target node, but we need to find previous on
                    // the bottom level.
                    unsafe {
                        prev = self.list.as_mut().as_mut().unwrap().get_prev_mut(next, 0);
                    }
                }
                break;
            }

            if level == 0 {
                break;
            }

            level -= 1;
        }

        (prev, next, found)
    }
}

impl Recycleable for Iter {
    fn new() -> Self {
        Iter {
            list: ptr::null_mut(),
            node: ptr::null_mut(),
            key: InternalKey {
                user_key: vec![],
                trailer: 0,
            },
            lower: vec![],
            upper: vec![],
        }
    }

    fn reset(&mut self) {
        self.list = ptr::null_mut();
        self.node = ptr::null_mut();
        self.lower.clear();
        self.upper.clear();
    }
}

impl InitializeWith<Iter> for Iter {
    fn initialize_with(&mut self, source: Iter) {
        *self = source
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::Arena;
    use crate::skl::{InternalKey, SKLError, Skiplist, INTERNAL_KEY_KIND_SET};

    #[test]
    fn test_seek_for_base_splice() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let iter = skl.iter(&[] as &[u8], &[] as &[u8]);
        let key1 = InternalKey::new(&[1u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key1, &[1u8]);

        let (prev, next, found) = iter.seek_for_base_splice(&[1u8]);
        assert!(!found);
        unsafe {
            assert_eq!([1u8], next.as_ref().get_key_mut(&mut a)[0..1]);
        }
        unsafe {
            assert_eq!(
                4u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                u64::from_ne_bytes(next.as_ref().get_key_mut(&mut a)[1..9].try_into().unwrap())
            )
        }
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

        let (prev, next, found) = iter.seek_for_base_splice(&[5u8]);
        assert!(!found);
        unsafe {
            assert_eq!([5u8], next.as_ref().get_key_mut(&mut a)[0..1]);
        }
        unsafe {
            assert_eq!(
                2u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                u64::from_ne_bytes(next.as_ref().get_key_mut(&mut a)[1..9].try_into().unwrap())
            )
        }
    }
}
