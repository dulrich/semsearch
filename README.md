# semsearch

Sync, in-process hybrid retrieval over one derived SQLite file: external-content FTS5 keyword search fused by reciprocal rank with a title-match leg, a document-level term-coverage leg and a caller-supplied vector leg.

```rust
use semsearch::{Chunk, Index, IndexOptions, Query};

let mut index = Index::open("search.db", &IndexOptions::default())?;
if index.rebuilt() { /* schema changed: re-feed every source */ }
index.upsert_source("notes/a.md", "hash-1", &[Chunk::new("c0", "Title", "Heading", "body text")])?;
for state in index.source_states()? { /* reconcile on state.content_hash */ }
let hits = index.search(&Query::new("body text"))?;
index.delete_source("notes/a.md")?;
let ro = Index::open_read_only("search.db", &IndexOptions::default())?; // never writes
```

## Install

Add it as a git dependency (there is no crates.io release):

```toml
semsearch = { git = "https://github.com/dulrich/semsearch", tag = "v0.2.1" }
```

## Scope

semsearch stores chunks and ranks them: bm25 keyword search, a title-match leg, a term-coverage leg and an optional vector leg, fused by reciprocal rank. It does not embed, chunk or fetch anything: there is no embedder, chunker or transport, so callers supply chunks and vectors. One recorded vector `Space` per index. The crate is synchronous, contains no `unsafe` code and needs no async runtime.

## The index is derived

The index file can always be rebuilt from its sources. A schema-version mismatch drops and recreates the file rather than migrating it; callers check `rebuilt()` after opening and re-feed every source when it is true.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests put their databases on tmpfs via `.cargo/config.toml`.

## Versioning

Semantic versioning. While 0.x, a public-API break bumps the minor version. A `SCHEMA_VERSION` bump is not an API break, because the index rebuilds and callers re-feed; the CHANGELOG notes it under the release. Each release is an annotated `vX.Y.Z` tag with a matching section in `CHANGELOG.md`.

## License

MIT, see `LICENSE`.
