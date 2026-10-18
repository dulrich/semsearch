# semsearch

Sync, in-process hybrid retrieval over one derived SQLite file: external-content FTS5 keyword search fused by reciprocal rank with a title-match leg and a document-level term-coverage leg (vector leg: WP-2).

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
