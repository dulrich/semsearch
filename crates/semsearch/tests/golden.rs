//! WP-R2 golden: full `Vec<Hit>` pinned for a mixed corpus (text-only, vector-only,
//! coverage-only and overlapping hits, leg ties, tied representative chunks). Synthetic data.

mod common;

use common::{TempDir, chunk, open, sources};
use semsearch::{Chunk, Hit, Index, Query, Space};

const DIM: usize = 8;

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[i] = 1.0;
    v
}

fn mix(a: usize, b: usize, wb: f32) -> Vec<f32> {
    let mut v = axis(a);
    v[b] = wb;
    v
}

fn meta(c: Chunk, m: &str) -> Chunk {
    Chunk {
        metadata: Some(m.to_owned()),
        ..c
    }
}

fn corpus() -> (TempDir, Index) {
    let dir = TempDir::new();
    let mut index = open(&dir);
    let space = Space::new("golden", DIM);
    let mut put = |source: &str, chunks: Vec<Chunk>| {
        index
            .upsert_source_in(&space, source, "h", &chunks)
            .expect("upsert");
    };
    // Text hits that tie on bm25 and title (same content), one with a far vector.
    put(
        "t-a",
        vec![meta(
            chunk("c", "Notes", "", "alpha beta here").with_embedding(axis(7)),
            "ta",
        )],
    );
    put("t-b", vec![chunk("c", "Notes", "", "alpha beta here")]);
    // Overlap: text + title + vector.
    put(
        "o-alpha",
        vec![
            chunk("c", "Alpha Beta", "Intro", "alpha beta overlap").with_embedding(mix(0, 1, 0.5)),
        ],
    );
    // Vector-only, tying on vector score.
    put(
        "v-b",
        vec![chunk("c", "Gamma", "", "gamma words").with_embedding(axis(0))],
    );
    put(
        "v-a",
        vec![chunk("c", "Delta", "", "delta words").with_embedding(axis(0))],
    );
    // Coverage-only with two tied chunks (same coverage count, same vector score).
    put(
        "k-two",
        vec![
            meta(
                chunk(
                    "first",
                    "Split One",
                    "H1",
                    "alpha lives in this first chunk",
                )
                .with_embedding(axis(2)),
                "m1",
            ),
            meta(
                chunk(
                    "second",
                    "Split Two",
                    "H2",
                    "beta lives in this second chunk",
                )
                .with_embedding(axis(2)),
                "m2",
            ),
        ],
    );
    // Coverage-only, no vectors, tying with k-two on coverage.
    put(
        "k-one",
        vec![
            chunk("x", "Other", "", "alpha only"),
            chunk("y", "Other", "", "beta only"),
        ],
    );
    // Partial: one term only (no coverage), text-unmatched.
    put("p", vec![chunk("c", "P", "", "alpha alone")]);
    (dir, index)
}

fn render(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| format!("{h:?}")).collect()
}

#[test]
fn golden_hybrid_and_keyword() {
    let (_dir, index) = corpus();
    let hybrid = render(
        &index
            .search(&Query::new("alpha beta").with_vector(axis(0)))
            .expect("s"),
    );
    let keyword = render(&index.search(&Query::new("alpha beta")).expect("s"));
    assert_eq!(hybrid, HYBRID);
    assert_eq!(keyword, KEYWORD);
    // Pure vector retrieval: the v-a/v-b tie breaks by source.
    let nearest = index.nearest_sources(&axis(0), 3, 64).expect("nearest");
    let names: Vec<&str> = nearest.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(names, ["v-a", "v-b", "o-alpha"]);
    let hits = index.search(&Query::new("gamma")).expect("s");
    assert_eq!(sources(&hits), ["v-b"]);
}

const HYBRID: &[&str] = &[
    r#"Hit { source: "o-alpha", chunk_id: "c", title: "Alpha Beta", heading: "Intro", score: 0.06505334374186833, score_text: Some(1.481559340280534), snippet: "alpha beta overlap", metadata: None, ranks: LegRanks { text: Some(1), title: Some(1), vector: Some(3), coverage: Some(1) } }"#,
    r#"Hit { source: "t-a", chunk_id: "c", title: "Notes", heading: "", score: 0.04788306451612903, score_text: Some(0.8115910214381992), snippet: "alpha beta here", metadata: Some("ta"), ranks: LegRanks { text: Some(2), title: None, vector: Some(4), coverage: Some(2) } }"#,
    r#"Hit { source: "t-b", chunk_id: "c", title: "Notes", heading: "", score: 0.031746031746031744, score_text: Some(0.8115910214381992), snippet: "alpha beta here", metadata: None, ranks: LegRanks { text: Some(3), title: None, vector: None, coverage: Some(3) } }"#,
    r#"Hit { source: "k-two", chunk_id: "first", title: "Split One", heading: "H1", score: 0.031009615384615385, score_text: None, snippet: "alpha lives in this first chunk", metadata: Some("m1"), ranks: LegRanks { text: None, title: None, vector: Some(5), coverage: Some(4) } }"#,
    r#"Hit { source: "v-a", chunk_id: "c", title: "Delta", heading: "", score: 0.01639344262295082, score_text: None, snippet: "delta words", metadata: None, ranks: LegRanks { text: None, title: None, vector: Some(1), coverage: None } }"#,
    r#"Hit { source: "v-b", chunk_id: "c", title: "Gamma", heading: "", score: 0.016129032258064516, score_text: None, snippet: "gamma words", metadata: None, ranks: LegRanks { text: None, title: None, vector: Some(2), coverage: None } }"#,
    r#"Hit { source: "k-one", chunk_id: "x", title: "Other", heading: "", score: 0.015384615384615385, score_text: None, snippet: "alpha only", metadata: None, ranks: LegRanks { text: None, title: None, vector: None, coverage: Some(5) } }"#,
];

const KEYWORD: &[&str] = &[
    r#"Hit { source: "o-alpha", chunk_id: "c", title: "Alpha Beta", heading: "Intro", score: 0.04918032786885246, score_text: Some(1.481559340280534), snippet: "alpha beta overlap", metadata: None, ranks: LegRanks { text: Some(1), title: Some(1), vector: None, coverage: Some(1) } }"#,
    r#"Hit { source: "t-a", chunk_id: "c", title: "Notes", heading: "", score: 0.03225806451612903, score_text: Some(0.8115910214381992), snippet: "alpha beta here", metadata: Some("ta"), ranks: LegRanks { text: Some(2), title: None, vector: None, coverage: Some(2) } }"#,
    r#"Hit { source: "t-b", chunk_id: "c", title: "Notes", heading: "", score: 0.031746031746031744, score_text: Some(0.8115910214381992), snippet: "alpha beta here", metadata: None, ranks: LegRanks { text: Some(3), title: None, vector: None, coverage: Some(3) } }"#,
    r#"Hit { source: "k-one", chunk_id: "x", title: "Other", heading: "", score: 0.015625, score_text: None, snippet: "alpha only", metadata: None, ranks: LegRanks { text: None, title: None, vector: None, coverage: Some(4) } }"#,
    r#"Hit { source: "k-two", chunk_id: "first", title: "Split One", heading: "H1", score: 0.015384615384615385, score_text: None, snippet: "alpha lives in this first chunk", metadata: Some("m1"), ranks: LegRanks { text: None, title: None, vector: None, coverage: Some(5) } }"#,
];
