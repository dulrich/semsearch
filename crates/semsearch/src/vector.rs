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

/// [`sign_code`] of [`from_blob`]`(b)`, written into `out` (resized to `b.len() / 4` dims), without
/// materialising the `f32` vector. Same predicate (`> 0.0`) and bit order.
pub(crate) fn sign_code_from_blob(b: &[u8], out: &mut Vec<u64>) {
    let chunks = b.as_chunks::<4>().0;
    out.clear();
    out.resize(chunks.len().div_ceil(64), 0);
    for (i, c) in chunks.iter().enumerate() {
        if f32::from_le_bytes(*c) > 0.0 {
            out[i / 64] |= 1 << (i % 64);
        }
    }
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

/// Outcome of [`Vectors::admit`]: whether a write in the checked space must record it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Admit {
    /// Nothing is recorded yet: the write must record its space.
    RecordSpace,
    /// The space is already recorded and equal.
    Recorded,
}

/// The index's vector state, owned together: the recorded space and the sign codes.
/// Invariant: codes are non-empty only under a recorded space, and the code word width is
/// `space.dim.div_ceil(64)`.
#[derive(Debug, Default)]
pub(crate) struct Vectors {
    space: Option<Space>,
    codes: Codes,
}

impl Vectors {
    /// Empty codes under `space` (as loaded at open); fill with [`Vectors::insert_loaded`].
    pub fn with_space(space: Option<Space>) -> Self {
        let codes = space
            .as_ref()
            .map_or_else(Codes::default, |s| Codes::new(s.dim));
        Self { space, codes }
    }

    /// Add one code while loading. Requires a recorded space.
    pub fn insert_loaded(&mut self, rowid: i64, code: &[u64]) {
        debug_assert!(self.space.is_some());
        self.codes.insert(rowid, code);
    }

    /// Forget the space and every code.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Validate `space` for a write against the recorded one.
    pub fn admit(&self, space: &Space) -> Result<Admit> {
        if space.dim == 0 || space.model.is_empty() {
            return Err(Error::InvalidInput(
                "space needs a model and dim > 0".into(),
            ));
        }
        match &self.space {
            None => Ok(Admit::RecordSpace),
            Some(recorded) if recorded == space => Ok(Admit::Recorded),
            Some(recorded) => Err(Error::SpaceMismatch {
                recorded: recorded.clone(),
                got: space.clone(),
            }),
        }
    }

    /// Apply a committed write: record `new_space` (fresh codes) if given, drop `removed`,
    /// then insert `added`.
    pub fn apply(
        &mut self,
        new_space: Option<Space>,
        removed: &[i64],
        added: Vec<(i64, Vec<u64>)>,
    ) {
        if let Some(space) = new_space {
            self.codes = Codes::new(space.dim);
            self.space = Some(space);
        }
        for &rowid in removed {
            self.codes.remove(rowid);
        }
        for (rowid, code) in added {
            self.codes.insert(rowid, &code);
        }
    }

    pub fn space(&self) -> Option<&Space> {
        self.space.as_ref()
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn nearest(&self, query: &[u64], pool: usize) -> Vec<i64> {
        self.codes.nearest(query, pool)
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
    fn sign_code_from_blob_matches_sign_code() {
        let edges = [
            0.0f32,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut buf = Vec::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for dim in [1usize, 8, 63, 64, 65, 130, 384] {
            let v: Vec<f32> = (0..dim)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if i % 5 == 0 {
                        edges[(seed % 8) as usize]
                    } else {
                        (seed as i32 as f32) / i32::MAX as f32
                    }
                })
                .collect();
            let blob = to_blob(&v);
            sign_code_from_blob(&blob, &mut buf);
            assert_eq!(buf, sign_code(&from_blob(&blob)), "dim {dim}");
        }
        let blob = to_blob(&edges);
        sign_code_from_blob(&blob, &mut buf);
        assert_eq!(buf, sign_code(&from_blob(&blob)));
        assert_eq!(buf, vec![0b0100_0100]);
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

    #[test]
    fn admit_first_same_mismatch() {
        let a = Space::new("m", 2);
        let mut v = Vectors::default();
        assert_eq!(v.admit(&a).expect("ok"), Admit::RecordSpace);
        assert!(v.admit(&Space::new("", 2)).is_err());
        assert!(v.admit(&Space::new("m", 0)).is_err());
        v.apply(Some(a.clone()), &[], vec![(1, sign_code(&[1.0, 1.0]))]);
        assert_eq!(v.len(), 1);
        assert_eq!(v.admit(&a).expect("ok"), Admit::Recorded);
        assert!(matches!(
            v.admit(&Space::new("m", 3)),
            Err(Error::SpaceMismatch { .. })
        ));
        v.reset();
        assert!(v.space().is_none() && v.len() == 0);
    }
}
