mod common;

use common::{TempDir, chunk, open, sources};
use rusqlite::Connection;
use semsearch::{Error, Index, IndexOptions, Query, SCHEMA_VERSION};

#[test]
fn fresh_open_builds_and_reopen_keeps() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    assert!(idx.rebuilt());
    idx.upsert_source("a.md", "h1", &[chunk("c0", "A", "", "alpha")])
        .unwrap();
    drop(idx);
    let idx = open(&dir);
    assert!(!idx.rebuilt());
    assert_eq!(idx.source_states().unwrap().len(), 1);
}

#[test]
fn version_mismatch_rebuilds_empty() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source("a.md", "h1", &[chunk("c0", "A", "", "alpha")])
        .unwrap();
    idx.set_meta("k", Some("v")).unwrap();
    drop(idx);
    let conn = Connection::open(dir.db()).unwrap();
    conn.execute(
        "UPDATE meta SET value = '999' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();
    drop(conn);
    let idx = open(&dir);
    assert!(idx.rebuilt());
    assert!(idx.source_states().unwrap().is_empty());
    assert_eq!(idx.get_meta("k").unwrap(), None);
    assert!(idx.search(&Query::new("alpha")).unwrap().is_empty());
}

#[test]
fn foreign_schema_is_dropped() {
    let dir = TempDir::new();
    let conn = Connection::open(dir.db()).unwrap();
    conn.execute_batch("CREATE TABLE chunks (x); CREATE VIRTUAL TABLE chunks_fts USING fts5(a);")
        .unwrap();
    drop(conn);
    let mut idx = open(&dir);
    assert!(idx.rebuilt());
    idx.upsert_source("a.md", "h", &[chunk("c0", "A", "", "alpha")])
        .unwrap();
    assert_eq!(
        sources(&idx.search(&Query::new("alpha")).unwrap()),
        ["a.md"]
    );
}

fn file_bytes(dir: &TempDir) -> Vec<u8> {
    std::fs::read(dir.db()).unwrap()
}

#[test]
fn read_only_open_never_mutates() {
    let dir = TempDir::new();
    // Missing file: error, file not created.
    let err = Index::open_read_only(dir.db(), &IndexOptions::default()).unwrap_err();
    assert!(matches!(err, Error::NotFound(_)));
    assert!(!dir.db().exists());

    // Wrong version: error, file untouched.
    let conn = Connection::open(dir.db()).unwrap();
    conn.execute_batch("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT); INSERT INTO meta VALUES ('schema_version', '0');")
        .unwrap();
    drop(conn);
    let before = file_bytes(&dir);
    let err = Index::open_read_only(dir.db(), &IndexOptions::default()).unwrap_err();
    assert!(
        matches!(err, Error::SchemaMismatch { found: Some(0), expected } if expected == SCHEMA_VERSION)
    );
    assert_eq!(file_bytes(&dir), before);

    // Good index: reads work, writes fail, content unchanged.
    std::fs::remove_file(dir.db()).unwrap();
    let mut idx = open(&dir);
    idx.upsert_source("a.md", "h1", &[chunk("c0", "A", "", "alpha")])
        .unwrap();
    drop(idx);
    let mut ro = Index::open_read_only(dir.db(), &IndexOptions::default()).unwrap();
    assert!(ro.is_read_only());
    assert_eq!(sources(&ro.search(&Query::new("alpha")).unwrap()), ["a.md"]);
    assert!(ro.upsert_source("b.md", "h", &[]).is_err());
    assert!(ro.delete_source("a.md").is_err());
    assert!(ro.set_meta("k", Some("v")).is_err());
    assert_eq!(ro.source_states().unwrap().len(), 1);
}

#[test]
fn upsert_replaces_atomically() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source(
        "a.md",
        "h1",
        &[
            chunk("c0", "A", "", "oldword"),
            chunk("c1", "A", "", "keep"),
        ],
    )
    .unwrap();
    idx.upsert_source("a.md", "h2", &[chunk("c0", "A", "", "newword")])
        .unwrap();
    let st = idx.source_states().unwrap();
    assert_eq!((st[0].content_hash.as_str(), st[0].chunks), ("h2", 1));
    assert!(idx.search(&Query::new("oldword")).unwrap().is_empty());
    assert!(idx.search(&Query::new("keep")).unwrap().is_empty());
    assert_eq!(
        sources(&idx.search(&Query::new("newword")).unwrap()),
        ["a.md"]
    );

    // A failing upsert (duplicate chunk id) leaves the previous state intact.
    let bad = [chunk("x", "A", "", "broken"), chunk("x", "A", "", "broken")];
    assert!(matches!(
        idx.upsert_source("a.md", "h3", &bad),
        Err(Error::InvalidInput(_))
    ));
    let st = idx.source_states().unwrap();
    assert_eq!((st[0].content_hash.as_str(), st[0].chunks), ("h2", 1));
    assert_eq!(
        sources(&idx.search(&Query::new("newword")).unwrap()),
        ["a.md"]
    );
}

#[test]
fn delete_removes_source_and_hits() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source("a.md", "h", &[chunk("c0", "A", "", "shared alpha")])
        .unwrap();
    idx.upsert_source("b.md", "h", &[chunk("c0", "B", "", "shared beta")])
        .unwrap();
    assert!(idx.delete_source("a.md").unwrap());
    assert!(!idx.delete_source("a.md").unwrap());
    assert_eq!(
        sources(&idx.search(&Query::new("shared")).unwrap()),
        ["b.md"]
    );
    assert!(idx.search(&Query::new("alpha")).unwrap().is_empty());
    // FTS external-content integrity holds after replace/delete.
    let conn = Connection::open(dir.db()).unwrap();
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)",
        [],
    )
    .unwrap();
}

#[test]
fn source_states_report_counts() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    idx.upsert_source(
        "b.md",
        "hb",
        &[chunk("c0", "B", "", "x"), chunk("c1", "B", "", "y")],
    )
    .unwrap();
    idx.upsert_source("a.md", "ha", &[]).unwrap();
    let st = idx.source_states().unwrap();
    let got: Vec<_> = st
        .iter()
        .map(|s| {
            (
                s.source.as_str(),
                s.content_hash.as_str(),
                s.chunks,
                s.embedded_chunks,
            )
        })
        .collect();
    assert_eq!(got, [("a.md", "ha", 0, 0), ("b.md", "hb", 2, 0)]);
}

#[test]
fn caller_meta_roundtrip() {
    let dir = TempDir::new();
    let mut idx = open(&dir);
    assert_eq!(idx.get_meta("import").unwrap(), None);
    idx.set_meta("import", Some("v1")).unwrap();
    idx.set_meta("import", Some("v2")).unwrap();
    assert_eq!(idx.get_meta("import").unwrap().as_deref(), Some("v2"));
    idx.set_meta("import", None).unwrap();
    assert_eq!(idx.get_meta("import").unwrap(), None);
}
