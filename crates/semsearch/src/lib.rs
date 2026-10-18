//! `semsearch` — sync hybrid retrieval over one derived SQLite/FTS5 file.
//!
//! One [`Index`] is one corpus. Callers feed it whole sources (a source is any caller id, e.g.
//! a vault-relative path) as lists of [`Chunk`]s; [`Index::search`] returns one [`Hit`] per
//! source, ranked by reciprocal-rank fusion of a weighted-bm25 leg, a title-match leg, an
//! optional vector leg (sign-bit Hamming pre-scan, exact f32 rescore) and a document-level
//! term-coverage leg. Vectors live in one recorded [`Space`]; see [`Index::upsert_source_in`].
//!
//! The database is derived: on a schema-version mismatch [`Index::open`] drops and recreates
//! it and reports [`Index::rebuilt`], so the caller re-feeds. [`Index::open_read_only`] never
//! creates, rebuilds or writes.

mod error;
mod fusion;
mod query;
mod search;
mod store;
mod vector;

pub use error::{Error, Result};
pub use search::{Hit, LegRanks, Query};
pub use store::{Chunk, Index, IndexOptions, SCHEMA_VERSION, SourceState};
pub use vector::{Space, VECTOR_POOL};
