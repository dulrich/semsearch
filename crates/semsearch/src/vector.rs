//! Vector primitives: embedding-space identity, normalization, LE f32 blobs, and the in-memory
//! sign-bit code table used for the Hamming pre-scan (plan Decision 5).

use std::collections::HashMap;

use crate::error::{Error, Result};

/// Default number of Hamming-scan candidates rescored with the exact f32 dot product.
pub const VECTOR_POOL: usize = 800;

/// One embedding space: a model reference plus its dimension. An index records exactly one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Space {
    pub model: String,
    pub dim: usize,
}

impl Space {
    pub fn new(model: &str, dim: usize) -> Self {
        Self {
            model: model.to_owned(),
            dim,
        }
    }
}

/// L2-normalize `v` (validated: expected dim, finite, non-zero).
pub(crate) fn normalized(v: &[f32], dim: usize) -> Result<Vec<f32>> {
    if v.len() != dim {
        return Err(Error::InvalidInput(format!(
            "vector has dim {}, space has dim {dim}",
            v.len()
        )));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidInput("vector has a non-finite value".into()));
    }
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(Error::InvalidInput("vector has zero norm".into()));
    }
    Ok(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
}

pub(crate) fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn from_blob(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One bit per dimension (`> 0.0`), packed into `u64` words.
pub(crate) fn sign_code(v: &[f32]) -> Vec<u64> {
    let mut out = vec![0u64; v.len().div_ceil(64)];
    for (i, x) in v.iter().enumerate() {
        if *x > 0.0 {
            out[i / 64] |= 1 << (i % 64);
        }
    }
    out
}

/// Sign codes of every embedded chunk, keyed by chunk rowid.
#[derive(Debug, Default)]
pub(crate) struct Codes {
    words: usize,
    rowids: Vec<i64>,
    bits: Vec<u64>,
    pos: HashMap<i64, usize>,
}

impl Codes {
    pub fn new(dim: usize) -> Self {
        Self {
            words: dim.div_ceil(64),
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.rowids.len()
    }

    /// Insert or replace one code.
    pub fn insert(&mut self, rowid: i64, code: &[u64]) {
        debug_assert_eq!(code.len(), self.words);
        if let Some(&p) = self.pos.get(&rowid) {
            self.bits[p * self.words..(p + 1) * self.words].copy_from_slice(code);
            return;
        }
        self.pos.insert(rowid, self.rowids.len());
        self.rowids.push(rowid);
        self.bits.extend_from_slice(code);
    }

    pub fn remove(&mut self, rowid: i64) {
        let Some(p) = self.pos.remove(&rowid) else {
            return;
        };
        let last = self.rowids.len() - 1;
        if p != last {
            let moved = self.rowids[last];
            self.rowids.swap(p, last);
            let w = self.words;
            self.bits.copy_within(last * w..(last + 1) * w, p * w);
            self.pos.insert(moved, p);
        }
        self.rowids.pop();
        self.bits.truncate(last * self.words);
    }

    /// The `pool` nearest rowids by Hamming distance (ties: lower rowid first).
    pub fn nearest(&self, query: &[u64], pool: usize) -> Vec<i64> {
        let w = self.words;
        let mut scored: Vec<(u32, i64)> = self
            .rowids
            .iter()
            .enumerate()
            .map(|(i, &rowid)| {
                let d = self.bits[i * w..(i + 1) * w]
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                (d, rowid)
            })
            .collect();
        if pool == 0 {
            return Vec::new();
        }
        if scored.len() > pool {
            scored.select_nth_unstable(pool - 1);
            scored.truncate(pool);
        }
        scored.into_iter().map(|(_, r)| r).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_is_unit_and_le_roundtrips() {
        let v = normalized(&[3.0, 4.0], 2).expect("ok");
        assert_eq!(v, vec![0.6, 0.8]);
        assert_eq!(from_blob(&to_blob(&v)), v);
        assert_eq!(to_blob(&[1.0])[..], 1.0f32.to_le_bytes());
        assert!(normalized(&[0.0, 0.0], 2).is_err());
        assert!(normalized(&[1.0], 2).is_err());
    }

    #[test]
    fn codes_insert_remove_nearest() {
        let mut c = Codes::new(2);
        c.insert(1, &sign_code(&[1.0, 1.0]));
        c.insert(2, &sign_code(&[-1.0, 1.0]));
        c.insert(3, &sign_code(&[-1.0, -1.0]));
        assert_eq!(c.nearest(&sign_code(&[1.0, 1.0]), 1), vec![1]);
        c.remove(1);
        assert_eq!(c.len(), 2);
        assert_eq!(c.nearest(&sign_code(&[1.0, 1.0]), 1), vec![2]);
        c.insert(3, &sign_code(&[1.0, 1.0]));
        assert_eq!(c.nearest(&sign_code(&[1.0, 1.0]), 1), vec![3]);
    }
}
