use crate::node::Node;
use crate::skl::{
    InternalKey, Skiplist, INTERNAL_KEY_KIND_INVALID, INTERNAL_KEY_KIND_MAX,
    INTERNAL_KEY_SEQ_NUM_MAX,
};
use lifeguard::{InitializeWith, Pool, Recycleable, Recycled};
use std::cell::RefCell;
use std::ptr::Unique;

thread_local!(pub static ITER_POOL:RefCell<Pool<Iter >> = RefCell::new(Pool::with_size_and_max(128, 2048)));

pub struct Iter {
    pub(crate) list: Unique<Skiplist>,
    pub(crate) node: Unique<Node>,
    pub(crate) key: InternalKey,
    pub(crate) lower: Vec<u8>,
    pub(crate) upper: Vec<u8>,
}

impl Iter {
    // Close resets the iterator.
    pub fn close(iter: Recycled<Iter>) {
        drop(iter)
    }

    pub fn seek_ge(
        &mut self,
        key: &[u8],
        try_seek_using_next: bool,
    ) -> Option<(Unique<InternalKey>, &[u8])> {
        if try_seek_using_next {
            unsafe {
                if self.node.as_ptr() == self.list.as_ref().tail.as_ptr() {
                    return None;
                }
            }
            let mut less = self.key.user_key.as_slice().lt(key);
            // Arbitrary constant. By measuring the seek cost as a function of the
            // number of elements in the skip list, and fitting to a model, we
            // could adjust the number of nexts based on the current size of the
            // skip list.
            let num_nexts = 5;
            let mut i = 0;
            while less && i < num_nexts {
                let res = self.next();
                res?;
                less = self.key.user_key.as_slice().lt(key);
                i += 1;
            }
            if !less {
                return Some((Unique::from(&mut self.key), self.value()));
            }
        }
        let (_, next, _) = self.seek_for_base_splice(key);
        self.node = next;
        unsafe {
            if self.node.as_ptr() == self.list.as_ref().tail.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.upper.is_empty() && self.upper.le(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().tail;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    pub fn seek_lt(&mut self, key: &[u8]) -> Option<(Unique<InternalKey>, &[u8])> {
        // NB: the top-level Iterator has already adjusted key based on
        // the upper-bound.
        let (prev, _, _) = self.seek_for_base_splice(key);
        self.node = prev;
        unsafe {
            if self.node.as_ptr() == self.list.as_ref().head.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.lower.is_empty() && self.lower.gt(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().head;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    pub fn first(&mut self) -> Option<(Unique<InternalKey>, &[u8])> {
        let skl = unsafe { self.list.as_mut() };
        self.node = skl.get_next_mut(skl.head, 0).unwrap();

        unsafe {
            if self.node.as_ptr() == self.list.as_ref().tail.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.upper.is_empty() && self.upper.le(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().tail;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    pub fn last(&mut self) -> Option<(Unique<InternalKey>, &[u8])> {
        let skl = unsafe { self.list.as_mut() };

        self.node = skl.get_prev_mut(skl.tail, 0).unwrap();

        unsafe {
            if self.node.as_ptr() == self.list.as_ref().head.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.lower.is_empty() && self.lower.gt(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().head;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(Unique<InternalKey>, &[u8])> {
        unsafe {
            self.node = self.list.as_mut().get_next_mut(self.node, 0).unwrap();
        }

        unsafe {
            if self.node.as_ptr() == self.list.as_ref().tail.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.upper.is_empty() && self.upper.le(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().tail;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    pub fn prev(&mut self) -> Option<(Unique<InternalKey>, &[u8])> {
        unsafe {
            self.node = self.list.as_mut().get_prev_mut(self.node, 0).unwrap();
        }

        unsafe {
            if self.node.as_ptr() == self.list.as_ref().head.as_ptr() {
                return None;
            }
        }
        self.decode_key();
        if !self.lower.is_empty() && self.lower.gt(&self.key.user_key) {
            unsafe {
                self.node = self.list.as_ref().head;
            }

            return None;
        }
        Some((Unique::from(&mut self.key), self.value()))
    }

    pub fn head(&self) -> bool {
        unsafe { self.node.as_ptr() == self.list.as_ref().head.as_ptr() }
    }

    pub fn tail(&self) -> bool {
        unsafe { self.node.as_ptr() == self.list.as_ref().tail.as_ptr() }
    }

    pub fn set_bounds(&mut self, lower: &[u8], upper: &[u8]) {
        self.lower = Vec::from(lower);
        self.upper = Vec::from(upper);
    }

    fn value(&mut self) -> &[u8] {
        unsafe {
            self.node
                .as_mut()
                .get_value_mut(self.list.as_mut().arena.as_mut())
        }
    }

    fn decode_key(&mut self) {
        let b = unsafe {
            self.list
                .as_mut()
                .arena
                .as_mut()
                .get_bytes_mut(self.node.as_ref().key_offset, self.node.as_ref().key_size)
        };
        // This is a manual inline of base.DecodeInternalKey, because the Go compiler
        // seems to refuse to automatically inline it currently.
        let l = b.len() as isize - 8;
        if l >= 0 {
            self.key.trailer = u64::from_ne_bytes(b[l as usize..].try_into().unwrap());
            self.key.user_key = Vec::from(&b[..l as usize]);
        } else {
            self.key.trailer = INTERNAL_KEY_KIND_INVALID as u64;
            self.key.user_key = vec![];
        }
    }

    fn seek_for_base_splice(&mut self, key: &[u8]) -> (Unique<Node>, Unique<Node>, bool) {
        let i_key = InternalKey::new(key, INTERNAL_KEY_SEQ_NUM_MAX, INTERNAL_KEY_KIND_MAX);
        let mut level = (unsafe { self.list.as_ref() }.height() - 1) as usize;

        let mut prev = unsafe { self.list.as_ref() }.head;
        let mut next: Unique<Node>;
        let mut found: bool;
        loop {
            unsafe {
                (prev, next, found) = self
                    .list
                    .as_mut()
                    .find_splice_for_level(&i_key, level, prev);
            }

            if found {
                if level != 0 {
                    // next is pointing at the target node, but we need to find previous on
                    // the bottom level.
                    unsafe {
                        prev = self.list.as_mut().get_prev_mut(next, 0).unwrap();
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
            list: Unique::dangling(),
            node: Unique::dangling(),
            key: InternalKey {
                user_key: vec![],
                trailer: 0,
            },
            lower: vec![],
            upper: vec![],
        }
    }

    fn reset(&mut self) {
        self.list = Unique::dangling();
        self.node = Unique::dangling();
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
    use crate::skl::{
        InternalKey, SKLError, Skiplist, INTERNAL_KEY_KIND_INVALID, INTERNAL_KEY_KIND_SET,
    };

    #[test]
    fn test_seek_for_base_splice() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let mut iter = skl.iter(&[] as &[u8], &[] as &[u8]);
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
        assert_eq!(prev.as_ptr(), skl.head.as_ptr());
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
            assert_eq!([4u8], prev.as_ref().get_key_mut(&mut a)[0..1]);
        }
        unsafe {
            assert_eq!(
                2u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                u64::from_ne_bytes(prev.as_ref().get_key_mut(&mut a)[1..9].try_into().unwrap())
            )
        }
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

    #[test]
    fn test_first_last() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let mut iter = skl.iter(&[] as &[u8], &[] as &[u8]);
        let key1 = InternalKey::new(&[1u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key1, &[1u8]);
        let key3 = InternalKey::new(&[1u8], 3u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key3, &[1u8]);
        let key4 = InternalKey::new(&[2u8], 5u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key4, &[1u8]);
        let key2 = InternalKey::new(&[2u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key2, &[2u8]);

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
        assert_eq!(prev.as_ptr(), skl.head.as_ptr());
        iter.decode_key();
        assert_eq!(INTERNAL_KEY_KIND_INVALID as u64, iter.key.trailer);

        let (k, v) = iter.first().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                4u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(1u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let (k, v) = iter.last().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                4u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(2u8, v[0]);
    }

    #[test]
    fn test_prev_next() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let mut iter = skl.iter(&[] as &[u8], &[] as &[u8]);
        let key1 = InternalKey::new(&[1u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key1, &[1u8]);
        let key3 = InternalKey::new(&[1u8], 3u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key3, &[1u8]);
        let key4 = InternalKey::new(&[2u8], 5u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key4, &[1u8]);
        let key2 = InternalKey::new(&[2u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key2, &[2u8]);

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
        assert_eq!(prev.as_ptr(), skl.head.as_ptr());
        iter.decode_key();
        assert_eq!(INTERNAL_KEY_KIND_INVALID as u64, iter.key.trailer);

        let (k, v) = iter.first().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                4u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(1u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let (k, v) = iter.next().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(1u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let (k, v) = iter.next().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                5u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let (k, v) = iter.last().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                4u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(2u8, v[0]);

        let (k, v) = iter.prev().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                5u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let (k, v) = iter.prev().unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(1u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);
    }

    #[test]
    fn test_seek() {
        let mut a = Arena::new(u32::MAX);

        let mut skl = Skiplist::new(&mut a).unwrap();
        let mut iter = skl.iter(&[] as &[u8], &[] as &[u8]);
        let key1 = InternalKey::new(&[1u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key1, &[1u8]);
        let key3 = InternalKey::new(&[1u8], 3u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key3, &[1u8]);
        let key4 = InternalKey::new(&[2u8], 5u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key4, &[1u8]);
        let key2 = InternalKey::new(&[2u8], 4u64, INTERNAL_KEY_KIND_SET);
        let _ = skl.add(&key2, &[2u8]);

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
        assert_eq!(prev.as_ptr(), skl.head.as_ptr());
        iter.decode_key();
        assert_eq!(INTERNAL_KEY_KIND_INVALID as u64, iter.key.trailer);

        let (k, v) = iter.seek_lt(&[2u8]).unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                3u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(1u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let res = iter.seek_lt(&[1u8]);
        assert!(res.is_none());

        let (k, v) = iter.seek_ge(&[2u8], true).unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                5u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let res = iter.seek_ge(&[3u8], true);
        assert!(res.is_none());

        let (k, v) = iter.seek_ge(&[2u8], false).unwrap();
        unsafe {
            assert_eq!(1, k.as_ref().user_key.len());
        }
        unsafe {
            assert_eq!(
                5u64 * 2u64.pow(8) + INTERNAL_KEY_KIND_SET as u64,
                k.as_ref().trailer
            );
        }
        unsafe {
            assert_eq!(2u8, k.as_ref().user_key[0]);
        }
        assert_eq!(1, v.len());
        assert_eq!(1u8, v[0]);

        let res = iter.seek_ge(&[3u8], false);
        assert!(res.is_none());

        assert!(iter.tail());
    }
}
