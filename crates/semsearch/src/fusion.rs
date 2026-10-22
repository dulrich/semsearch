//! N-leg reciprocal-rank fusion (port of `fuseSearchRows`, ranking.mjs:224).
//!
//! Candidates are indexed `0..n`. The text (bm25) leg is given as a rank per candidate; every
//! other leg is a weight plus an optional score per candidate (higher is better). A leg ranks
//! its scored candidates by score descending, ties broken by text rank (absent text rank sorts
//! last), then by candidate index. An empty leg contributes nothing.

/// RRF constant (`RRF_K`, ranking.mjs:16).
pub(crate) const RRF_K: f64 = 60.0;

/// One non-text ranking leg.
pub(crate) struct Leg {
    pub weight: f64,
    /// `scores[i]` is candidate `i`'s score on this leg; `None` keeps it out of the leg.
    pub scores: Vec<Option<f64>>,
}

/// One fused candidate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fused {
    pub index: usize,
    pub rrf: f64,
    /// 1-based rank per leg, same order as the `legs` argument.
    pub leg_ranks: Vec<Option<usize>>,
}

/// Fuse. `text_ranks[i]` is candidate `i`'s 1-based bm25 rank (weight 1.0) or `None`.
/// Returns every candidate, sorted by fused score descending.
pub(crate) fn fuse(text_ranks: &[Option<usize>], legs: &[Leg]) -> Vec<Fused> {
    let n = text_ranks.len();
    let text_order = |i: usize| text_ranks[i].unwrap_or(usize::MAX);
    let mut out: Vec<Fused> = (0..n)
        .map(|i| Fused {
            index: i,
            rrf: text_ranks[i].map_or(0.0, |r| 1.0 / (RRF_K + r as f64)),
            leg_ranks: vec![None; legs.len()],
        })
        .collect();
    for (leg_no, leg) in legs.iter().enumerate() {
        let mut scored: Vec<(usize, f64)> = (0..n)
            .filter_map(|i| leg.scores.get(i).copied().flatten().map(|s| (i, s)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| text_order(a.0).cmp(&text_order(b.0)))
                .then_with(|| a.0.cmp(&b.0))
        });
        for (rank0, (i, _)) in scored.into_iter().enumerate() {
            let rank = rank0 + 1;
            out[i].leg_ranks[leg_no] = Some(rank);
            out[i].rrf += leg.weight / (RRF_K + rank as f64);
        }
    }
    out.sort_by(|a, b| {
        b.rrf
            .total_cmp(&a.rrf)
            .then_with(|| text_order(a.index).cmp(&text_order(b.index)))
            .then_with(|| a.index.cmp(&b.index))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_rrf_fusion() {
        // Three candidates: 0 and 1 from bm25 (ranks 1, 2); 2 is leg-only.
        let text = [Some(1), Some(2), None];
        let legs = [
            Leg {
                weight: 1.0,
                scores: vec![None, Some(0.9), Some(0.9)],
            },
            Leg {
                weight: 1.0,
                scores: vec![Some(0.5), Some(1.0), None],
            },
        ];
        let fused = fuse(&text, &legs);
        let order: Vec<usize> = fused.iter().map(|f| f.index).collect();
        assert_eq!(order, vec![1, 0, 2]);
        // Candidate 1: text 2, leg0 rank 1 (tie broken by text rank), leg1 rank 1.
        assert_eq!(fused[0].leg_ranks, vec![Some(1), Some(1)]);
        let expected = 1.0 / 62.0 + 1.0 / 61.0 + 1.0 / 61.0;
        assert!((fused[0].rrf - expected).abs() < 1e-12);
        // Candidate 0: text 1, leg1 rank 2.
        assert!((fused[1].rrf - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        // Candidate 2: leg0 rank 2 only.
        assert_eq!(fused[2].leg_ranks, vec![Some(2), None]);
        assert!((fused[2].rrf - 1.0 / 62.0).abs() < 1e-12);
        // Same input, same output.
        assert_eq!(fuse(&text, &legs), fused);
    }

    #[test]
    fn no_legs_preserves_text_order() {
        let fused = fuse(&[Some(2), Some(1), None], &[]);
        let order: Vec<usize> = fused.iter().map(|f| f.index).collect();
        assert_eq!(order, vec![1, 0, 2]);
    }
}
