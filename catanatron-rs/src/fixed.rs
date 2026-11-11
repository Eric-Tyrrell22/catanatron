//! `FixedVec`: an inline, fixed-capacity vector. Lets `State` hold its
//! variable-length lists without heap allocations, so copying a state is a
//! plain memory copy.

use std::fmt;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy)]
pub struct FixedVec<T: Copy + Default, const N: usize> {
    len: u8,
    items: [T; N],
}

impl<T: Copy + Default, const N: usize> FixedVec<T, N> {
    pub fn new() -> Self {
        assert!(N <= u8::MAX as usize, "FixedVec capacity must fit in u8");
        FixedVec { len: 0, items: [T::default(); N] }
    }

    /// `n` copies of `value`.
    pub fn filled(value: T, n: usize) -> Self {
        let mut v = Self::new();
        for _ in 0..n {
            v.push(value);
        }
        v
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        assert!((self.len as usize) < N, "FixedVec capacity {N} exceeded");
        self.items[self.len as usize] = value;
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.items[self.len as usize])
    }

    /// Removes and returns the item at `index`, shifting later items left.
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len as usize;
        assert!(index < len, "FixedVec::remove index out of bounds");
        let value = self.items[index];
        self.items.copy_within(index + 1..len, index);
        self.len -= 1;
        value
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut w = 0;
        for r in 0..self.len as usize {
            if keep(&self.items[r]) {
                self.items[w] = self.items[r];
                w += 1;
            }
        }
        self.len = w as u8;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<T: Copy + Default, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> Deref for FixedVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        &self.items[..self.len as usize]
    }
}

impl<T: Copy + Default, const N: usize> DerefMut for FixedVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len as usize]
    }
}

impl<T: Copy + Default, const N: usize> FromIterator<T> for FixedVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        for x in iter {
            v.push(x);
        }
        v
    }
}

impl<T: Copy + Default + PartialEq, const N: usize> PartialEq for FixedVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Copy + Default + Eq, const N: usize> Eq for FixedVec<T, N> {}

impl<T: Copy + Default + fmt::Debug, const N: usize> fmt::Debug for FixedVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: Copy + Default, const N: usize> IntoIterator for &'a FixedVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
