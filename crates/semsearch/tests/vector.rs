//! WP-2: embedding space, sign codes, vector leg, hybrid fusion. Synthetic data only.

mod common;

use common::{TempDir, chunk, open, sources};
use semsearch::{Error, Index, IndexOptions, Query, Space};

const DIM: usize = 64;

fn space() -> Space {
    Space::new("test-model", DIM)
}

/// `n` single-chunk sources `s000..`, each with text "common word{i}" and a random vector.
fn corpus(index: &mut Index, n: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for i in 0..n {
        let source = format!("s{i:03}");
        let v = rng.vector(DIM);
        let c = chunk("c0", &format!("Doc {i}"), "", &format!("common word{i}"))
            .with_embedding(v.clone());
        index
            .upsert_source_in(&space(), &source, "h", &[c])
            .expect("upsert");
        out.push((source, unit(&v)));
    }
    out
}

fn exact_order(corpus: &[(String, Vec<f32>)], q: &[f32]) -> Vec<String> {
    let q = unit(q);
    let mut scored: Vec<(f32, &String)> = corpus.iter().map(|(s, v)| (dot(v, &q), s)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().map(|(_, s)| s.clone()).collect()
}

#[test]
fn first_write_records_space_and_mismatch_is_rejected() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    assert_eq!(index.space(), None);
    // Keyword-only writes record nothing.
    index
        .upsert_source("a", "h", &[chunk("c", "A", "", "alpha")])
        .expect("upsert");
    assert_eq!(index.space(), None);
    // An embedding without a space is rejected.
    let err = index
        .upsert_source(
            "b",
            "h",
            &[chunk("c", "B", "", "beta").with_embedding(vec![1.0; DIM])],
        )
        .expect_err("needs space");
    assert!(matches!(err, Error::InvalidInput(_)));
    corpus(&mut index, 2, 1);
    assert_eq!(index.space(), Some(&space()));
    for other in [Space::new("other-model", DIM), Space::new("test-model", 32)] {
        let c = chunk("c", "X", "", "x").with_embedding(vec![1.0; other.dim]);
        let err = index
            .upsert_source_in(&other, "x", "h", &[c])
            .expect_err("mismatch");
        assert!(matches!(err, Error::SpaceMismatch { .. }));
        let err = index
            .set_embeddings(&other, "a", &[("c".into(), vec![1.0; other.dim])])
            .expect_err("mismatch");
        assert!(matches!(err, Error::SpaceMismatch { .. }));
    }
    // Wrong dim within the right space, and nothing was written by any rejected call.
    let c = chunk("c", "X", "", "x").with_embedding(vec![1.0; DIM + 1]);
    assert!(index.upsert_source_in(&space(), "x", "h", &[c]).is_err());
    let states = index.source_states().expect("states");
    assert_eq!(
        states.iter().map(|s| s.source.as_str()).collect::<Vec<_>>(),
        vec!["a", "s000", "s001"]
    );
    assert_eq!(states[0].embedded_chunks, 0);
    drop(index);
    let reopened = Index::open_read_only(dir.db(), &IndexOptions::default()).expect("ro");
    assert_eq!(reopened.space(), Some(&space()));
    assert_eq!(reopened.embedded_count(), 2);
}

#[test]
fn reset_space_allows_a_new_space() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    corpus(&mut index, 3, 2);
    index.reset_space().expect("reset");
    assert_eq!(index.space(), None);
    assert_eq!(index.embedded_count(), 0);
    assert!(
        index
            .source_states()
            .expect("states")
            .iter()
            .all(|s| s.embedded_chunks == 0)
    );
    let other = Space::new("other-model", 8);
    index
        .set_embeddings(&other, "s000", &[("c0".into(), vec![1.0; 8])])
        .expect("new space");
    assert_eq!(index.space(), Some(&other));
    assert_eq!(index.embedded_count(), 1);
}

#[test]
fn vectors_are_normalized_and_bad_vectors_rejected() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let mut v = vec![0.0; DIM];
    v[0] = 3.0;
    v[1] = 4.0;
    index
        .upsert_source_in(
            &space(),
            "a",
            "h",
            &[chunk("c", "A", "", "alpha").with_embedding(v)],
        )
        .expect("upsert");
    // A unit query along dim 0 scores exactly 0.6 only if the stored vector is normalized;
    // the hit's vector rank proves it scored, and an unnormalized query scores the same.
    let mut q = vec![0.0; DIM];
    q[0] = 10.0;
    let hits = index
        .search(&Query::new("alpha").with_vector(q))
        .expect("search");
    assert_eq!(hits[0].ranks.vector, Some(1));
    for bad in [vec![0.0; DIM], {
        let mut b = vec![1.0; DIM];
        b[3] = f32::NAN;
        b
    }] {
        let err = index
            .upsert_source_in(
                &space(),
                "b",
                "h",
                &[chunk("c", "B", "", "b").with_embedding(bad)],
            )
            .expect_err("bad vector");
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}

#[test]
fn codes_track_every_write_path_and_reopen() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    corpus(&mut index, 5, 3);
    assert_eq!(index.embedded_count(), 5);
    // Replace with two embedded chunks and one bare chunk.
    let mut rng = Rng::new(99);
    index
        .upsert_source_in(
            &space(),
            "s000",
            "h2",
            &[
                chunk("a", "A", "", "x").with_embedding(rng.vector(DIM)),
                chunk("b", "B", "", "y").with_embedding(rng.vector(DIM)),
                chunk("c", "C", "", "z"),
            ],
        )
        .expect("replace");
    assert_eq!(index.embedded_count(), 6);
    assert!(index.delete_source("s001").expect("delete"));
    assert_eq!(index.embedded_count(), 5);
    index
        .set_embeddings(&space(), "s000", &[("c".into(), rng.vector(DIM))])
        .expect("backfill");
    assert_eq!(index.embedded_count(), 6);
    // Re-embedding an already-embedded chunk replaces, not adds.
    index
        .set_embeddings(&space(), "s000", &[("a".into(), rng.vector(DIM))])
        .expect("re-embed");
    assert_eq!(index.embedded_count(), 6);
    // Keyword-only upsert drops that source's codes.
    index
        .upsert_source("s002", "h3", &[chunk("c0", "D", "", "w")])
        .expect("kw upsert");
    assert_eq!(index.embedded_count(), 5);
    let total: u64 = index
        .source_states()
        .expect("states")
        .iter()
        .map(|s| s.embedded_chunks)
        .sum();
    assert_eq!(total, 5);
    drop(index);
    let reopened = open(&dir);
    assert_eq!(reopened.embedded_count(), 5);
}

#[test]
fn set_embeddings_rejects_unknown_source_or_chunk_atomically() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    index
        .upsert_source(
            "a",
            "h",
            &[chunk("c1", "A", "", "alpha"), chunk("c2", "A", "", "beta")],
        )
        .expect("upsert");
    let v = vec![1.0; DIM];
    assert!(matches!(
        index.set_embeddings(&space(), "nope", &[("c1".into(), v.clone())]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        index.set_embeddings(
            &space(),
            "a",
            &[("c1".into(), v.clone()), ("zz".into(), v.clone())]
        ),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(index.space(), None);
    assert_eq!(index.embedded_count(), 0);
    assert_eq!(index.source_states().expect("states")[0].embedded_chunks, 0);
}

#[test]
fn vector_leg_finds_nearest_neighbour() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let docs = corpus(&mut index, 50, 4);
    // Query = a small perturbation of s017's vector.
    let mut q = docs[17].1.clone();
    q[0] += 0.05;
    let hits = index
        .search(&Query::new("common").with_limit(100).with_vector(q))
        .expect("search");
    let top = hits
        .iter()
        .find(|h| h.ranks.vector == Some(1))
        .expect("vector rank 1");
    assert_eq!(top.source, "s017");
}

#[test]
fn full_pool_recall_equals_exact_and_small_pool_caps_leg() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let docs = corpus(&mut index, 60, 5);
    let mut rng = Rng::new(77);
    for _ in 0..5 {
        let q = rng.vector(DIM);
        let exact = exact_order(&docs, &q);
        let hits = index
            .search(
                &Query::new("common")
                    .with_limit(100)
                    .with_vector(q.clone())
                    .with_vector_pool(60),
            )
            .expect("search");
        assert_eq!(hits.len(), 60);
        for h in &hits {
            let r = h.ranks.vector.expect("every source in the leg");
            assert_eq!(exact[r - 1], h.source);
        }
        let small = index
            .search(
                &Query::new("common")
                    .with_limit(100)
                    .with_vector(q)
                    .with_vector_pool(5),
            )
            .expect("search");
        assert_eq!(small.iter().filter(|h| h.ranks.vector.is_some()).count(), 5);
    }
}

#[test]
fn no_query_vector_equals_keyword_output() {
    let plain_dir = TempDir::new();
    let mut plain = open(&plain_dir);
    let vec_dir = TempDir::new();
    let mut with_vectors = open(&vec_dir);
    corpus(&mut with_vectors, 30, 6);
    for i in 0..30 {
        plain
            .upsert_source(
                &format!("s{i:03}"),
                "h",
                &[chunk(
                    "c0",
                    &format!("Doc {i}"),
                    "",
                    &format!("common word{i}"),
                )],
            )
            .expect("upsert");
    }
    for q in ["common", "common word7", "doc word1"] {
        let a = plain.search(&Query::new(q).with_limit(50)).expect("plain");
        let b = with_vectors
            .search(&Query::new(q).with_limit(50))
            .expect("vectors");
        assert_eq!(a, b);
        assert!(b.iter().all(|h| h.ranks.vector.is_none()));
        // A query vector on an index with no space skips the leg: still identical.
        let c = plain
            .search(&Query::new(q).with_limit(50).with_vector(vec![1.0; DIM]))
            .expect("no space");
        assert_eq!(a, c);
    }
}

#[test]
fn query_vector_dim_mismatch_is_an_error() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    corpus(&mut index, 3, 7);
    let err = index
        .search(&Query::new("common").with_vector(vec![1.0; DIM - 1]))
        .expect_err("dim");
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn named_leg_ranks_with_vector_and_no_coverage() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let docs = corpus(&mut index, 10, 8);
    // Single term: coverage does not apply; title leg matches "Doc" titles of all sources.
    let hits = index
        .search(
            &Query::new("doc")
                .with_limit(20)
                .with_vector(docs[3].1.clone()),
        )
        .expect("search");
    assert!(hits.iter().all(|h| h.ranks.coverage.is_none()));
    assert!(
        hits.iter()
            .all(|h| h.ranks.title.is_some() && h.ranks.vector.is_some())
    );
    let s3 = hits.iter().find(|h| h.source == "s003").expect("s003");
    assert_eq!(s3.ranks.vector, Some(1));
    // Title ranks are unaffected by the vector leg: same as the keyword-only search.
    let kw = index.search(&Query::new("doc").with_limit(20)).expect("kw");
    for h in &hits {
        let k = kw
            .iter()
            .find(|k| k.source == h.source)
            .expect("same sources");
        assert_eq!(h.ranks.title, k.ranks.title);
        assert_eq!(h.ranks.text, k.ranks.text);
    }
}

#[test]
fn vector_only_hit_is_hydrated() {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let mut rng = Rng::new(9);
    let target = rng.vector(DIM);
    let text = "an unrelated passage about gardens and weather patterns";
    index
        .upsert_source_in(
            &space(),
            "vec-only",
            "h",
            &[chunk("c0", "Garden", "Intro", text).with_embedding(target.clone())],
        )
        .expect("upsert");
    let negated: Vec<f32> = target.iter().map(|x| -x).collect();
    index
        .upsert_source_in(
            &space(),
            "kw",
            "h",
            &[chunk("c0", "Rust", "", "rust ownership").with_embedding(negated)],
        )
        .expect("upsert");
    let hits = index
        .search(&Query::new("rust").with_vector(target))
        .expect("search");
    assert_eq!(sources(&hits).len(), 2);
    let v = hits
        .iter()
        .find(|h| h.source == "vec-only")
        .expect("vector-only hit");
    assert_eq!(v.score_text, None);
    assert_eq!(v.ranks.text, None);
    assert_eq!(v.ranks.vector, Some(1));
    assert_eq!(v.chunk_id, "c0");
    assert_eq!(v.title, "Garden");
    assert_eq!(v.heading, "Intro");
    assert!(!v.snippet.is_empty() && text.starts_with(v.snippet.trim_end_matches("...").trim()));
    let k = hits.iter().find(|h| h.source == "kw").expect("kw hit");
    assert_eq!((k.ranks.text, k.ranks.vector), (Some(1), Some(2)));
}

/// Deterministic xorshift vectors (unnormalized, components in [-1, 1)).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
