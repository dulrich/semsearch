mod common;

use common::{TempDir, chunk, open, sources};
use semsearch::{Chunk, Query};

#[test]
fn title_outweighs_heading_outweighs_text() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    // Distinct sources so the title-match leg (path/title) can't help any of them unequally:
    // only `t.md` has the word in its title, which both bm25 and the title leg favor.
    idx.upsert_source(
        "x1.md",
        "h",
        &[chunk("c", "Other", "", "zephyr filler words here")],
    )
    .unwrap();
    idx.upsert_source(
        "x2.md",
        "h",
        &[chunk("c", "Other", "Zephyr", "filler words here")],
    )
    .unwrap();
    idx.upsert_source(
        "x3.md",
        "h",
        &[chunk("c", "Zephyr", "", "filler words here")],
    )
    .unwrap();
    let hits = idx.search(&Query::new("zephyr")).unwrap();
    assert_eq!(sources(&hits), ["x3.md", "x2.md", "x1.md"]);
    assert_eq!(hits[0].ranks.text, Some(1));
    assert_eq!(hits[1].ranks.text, Some(2));
    assert!(hits[0].ranks.title.is_some());
    assert!(hits[1].ranks.title.is_none());
    assert!(hits[0].score_text.unwrap() > hits[1].score_text.unwrap());
}

#[test]
fn heading_outweighs_text_on_bm25() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source("a.md", "h", &[chunk("c", "Other", "", "quokka filler")])
        .unwrap();
    idx.upsert_source("b.md", "h", &[chunk("c", "Other", "Quokka", "filler")])
        .unwrap();
    assert_eq!(
        sources(&idx.search(&Query::new("quokka")).unwrap()),
        ["b.md", "a.md"]
    );
}

#[test]
fn entities_are_searchable() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    let mut c = chunk("c", "Paper", "", "about things");
    c.entities = "Ada Lovelace".into();
    c.metadata = Some("{\"k\":1}".into());
    idx.upsert_source("p.md", "h", &[c]).unwrap();
    let hits = idx.search(&Query::new("lovelace")).unwrap();
    assert_eq!(sources(&hits), ["p.md"]);
    assert_eq!(hits[0].metadata.as_deref(), Some("{\"k\":1}"));
}

#[test]
fn prefix_applies_to_last_token_only_when_long_enough() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source("a.md", "h", &[chunk("c", "T", "", "gardening tomatoes")])
        .unwrap();
    // Last token >= 3 chars: prefix match.
    assert_eq!(sources(&idx.search(&Query::new("gar")).unwrap()), ["a.md"]);
    assert_eq!(
        sources(&idx.search(&Query::new("tomatoes gard")).unwrap()),
        ["a.md"]
    );
    // Last token < 3 chars: exact only.
    assert!(idx.search(&Query::new("ga")).unwrap().is_empty());
    // Non-last tokens are never prefix-expanded.
    assert!(idx.search(&Query::new("gard xyzzy")).unwrap().is_empty());
}

#[test]
fn coverage_favors_multi_term_documents() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    // `multi.md` holds both terms but in different chunks: the per-chunk AND misses it,
    // coverage finds it. `single*.md` hold only one term each.
    idx.upsert_source(
        "multi.md",
        "h",
        &[
            chunk("c0", "M", "", "walrus facts"),
            chunk("c1", "M", "", "narwhal facts"),
        ],
    )
    .unwrap();
    for i in 0..3 {
        idx.upsert_source(
            &format!("single{i}.md"),
            "h",
            &[chunk("c", "S", "", "walrus only")],
        )
        .unwrap();
    }
    let hits = idx.search(&Query::new("walrus narwhal")).unwrap();
    assert_eq!(sources(&hits), ["multi.md"]);
    let hit = &hits[0];
    assert_eq!(hit.ranks.text, None);
    assert_eq!(hit.ranks.coverage, Some(1));
    assert_eq!(hit.score_text, None);
    assert!(hit.snippet.contains("facts"));

    // When FTS also matches, the fully-covering doc outranks a partial-coverage doc.
    idx.upsert_source("both.md", "h", &[chunk("c", "B", "", "walrus narwhal")])
        .unwrap();
    let hits = idx.search(&Query::new("walrus narwhal")).unwrap();
    assert_eq!(hits[0].source, "both.md");
    assert_eq!(hits[0].ranks.coverage, Some(1));
}

#[test]
fn one_hit_per_source_and_limit() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    let chunks: Vec<Chunk> = (0..5)
        .map(|i| chunk(&format!("c{i}"), "T", "", "lemur"))
        .collect();
    idx.upsert_source("a.md", "h", &chunks).unwrap();
    for i in 0..5 {
        idx.upsert_source(&format!("s{i}.md"), "h", &[chunk("c", "T", "", "lemur")])
            .unwrap();
    }
    assert_eq!(idx.search(&Query::new("lemur")).unwrap().len(), 6);
    assert_eq!(
        idx.search(&Query::new("lemur").with_limit(2))
            .unwrap()
            .len(),
        2
    );
    assert!(idx.search(&Query::new("  !! ")).unwrap().is_empty());
}

#[test]
fn search_is_deterministic_and_snippets_present() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    for i in 0..10 {
        idx.upsert_source(
            &format!("n{i}.md"),
            "h",
            &[chunk(
                "c",
                &format!("Note {i}"),
                "",
                "otter river otter stream",
            )],
        )
        .unwrap();
    }
    let a = idx.search(&Query::new("otter stream")).unwrap();
    let b = idx.search(&Query::new("otter stream")).unwrap();
    assert_eq!(a, b);
    assert!(a.iter().all(|h| h.snippet.contains("otter")));
}
