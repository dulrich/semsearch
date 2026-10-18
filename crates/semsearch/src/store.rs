//! Storage: open/rebuild, schema, source writes, meta.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};

use crate::error::{Error, Result};

/// Schema version stored in `meta`. A mismatch rebuilds an empty index; never migrates.
pub const SCHEMA_VERSION: i64 = 1;

const META_SCHEMA_VERSION: &str = "schema_version";
/// Reserved for WP-2: embedding-space identity (nullable until vectors exist).
const META_EMBEDDING_MODEL: &str = "embedding_model";
const META_EMBEDDING_DIM: &str = "embedding_dim";

const SCHEMA_SQL: &str = r"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;
CREATE TABLE user_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE sources (id TEXT PRIMARY KEY, content_hash TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL REFERENCES sources(id),
  chunk_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  title TEXT NOT NULL,
  heading TEXT NOT NULL,
  text TEXT NOT NULL,
  entities TEXT NOT NULL,
  metadata TEXT,
  embedding BLOB,
  UNIQUE (source, chunk_id)
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  title, heading, text, entities,
  content='chunks', content_rowid='id', prefix='3'
);
";

/// Open options.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// SQLite `busy_timeout`, applied before any other pragma.
    pub busy_timeout: Duration,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            busy_timeout: Duration::from_secs(5),
        }
    }
}

/// One chunk of a source, as supplied by the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Chunk {
    /// Chunk id, unique within its source.
    pub id: String,
    /// Position within the source.
    pub ordinal: i64,
    pub title: String,
    pub heading: String,
    pub text: String,
    /// Free-text entity facet (authors, tags, ...), indexed with weight 8.
    pub entities: String,
    /// Opaque caller payload, returned on hits.
    pub metadata: Option<String>,
}

impl Chunk {
    /// Convenience constructor; `ordinal`, `entities`, `metadata` default.
    pub fn new(id: &str, title: &str, heading: &str, text: &str) -> Self {
        Self {
            id: id.to_owned(),
            title: title.to_owned(),
            heading: heading.to_owned(),
            text: text.to_owned(),
            ..Self::default()
        }
    }
}

/// Per-source state for the caller's reconcile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceState {
    pub source: String,
    pub content_hash: String,
    pub chunks: u64,
    /// Chunks carrying an embedding (always 0 until WP-2).
    pub embedded_chunks: u64,
}

/// One search index over one SQLite file.
#[derive(Debug)]
pub struct Index {
    pub(crate) conn: Connection,
    rebuilt: bool,
    read_only: bool,
}

impl Index {
    /// Open or create. On a missing schema or a version mismatch the file's tables are dropped
    /// and recreated empty, and [`Index::rebuilt`] returns `true`.
    pub fn open(path: impl AsRef<Path>, opts: &IndexOptions) -> Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        conn.busy_timeout(opts.busy_timeout)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let mut index = Self {
            conn,
            rebuilt: false,
            read_only: false,
        };
        if index.schema_version()? != Some(SCHEMA_VERSION) {
            index.rebuild()?;
            index.rebuilt = true;
        }
        Ok(index)
    }

    /// Open an existing index read-only. Never creates, rebuilds or writes: a missing file is
    /// [`Error::NotFound`], a wrong or absent schema version is [`Error::SchemaMismatch`].
    pub fn open_read_only(path: impl AsRef<Path>, opts: &IndexOptions) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::NotFound(path.to_path_buf()));
        }
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(opts.busy_timeout)?;
        let index = Self {
            conn,
            rebuilt: false,
            read_only: true,
        };
        let found = index.schema_version()?;
        if found != Some(SCHEMA_VERSION) {
            return Err(Error::SchemaMismatch {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(index)
    }

    /// Whether [`Index::open`] dropped and recreated the index (caller must re-feed).
    pub fn rebuilt(&self) -> bool {
        self.rebuilt
    }

    /// Whether this handle was opened with [`Index::open_read_only`].
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn schema_version(&self) -> Result<Option<i64>> {
        let has_meta: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
            [],
            |r| r.get(0),
        )?;
        if !has_meta {
            return Ok(None);
        }
        let value: Option<Option<String>> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [META_SCHEMA_VERSION],
                |r| r.get(0),
            )
            .optional()?;
        Ok(value.flatten().and_then(|v| v.parse().ok()))
    }

    fn rebuild(&mut self) -> Result<()> {
        // Drop order across foreign keys of an unknown old schema is not ours to resolve.
        let fk: bool = self
            .conn
            .pragma_query_value(None, "foreign_keys", |r| r.get(0))?;
        self.conn.pragma_update(None, "foreign_keys", false)?;
        let result = self.rebuild_tables();
        self.conn.pragma_update(None, "foreign_keys", fk)?;
        result
    }

    fn rebuild_tables(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        // Virtual tables first: dropping one drops its shadow tables.
        for virtual_first in [true, false] {
            let names: Vec<String> = {
                let mut stmt = tx.prepare(
                    "SELECT name FROM sqlite_master WHERE type = 'table' \
                     AND name NOT LIKE 'sqlite_%' \
                     AND (sql LIKE 'CREATE VIRTUAL TABLE%') = ?1",
                )?;
                stmt.query_map([virtual_first], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for name in names {
                tx.execute_batch(&format!(
                    "DROP TABLE IF EXISTS \"{}\";",
                    name.replace('"', "\"\"")
                ))?;
            }
        }
        tx.execute_batch(SCHEMA_SQL)?;
        tx.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2), (?3, NULL), (?4, NULL)",
            params![
                META_SCHEMA_VERSION,
                SCHEMA_VERSION.to_string(),
                META_EMBEDDING_MODEL,
                META_EMBEDDING_DIM
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Atomically replace every chunk of `source` (creating the source if new).
    pub fn upsert_source(
        &mut self,
        source: &str,
        content_hash: &str,
        chunks: &[Chunk],
    ) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for chunk in chunks {
            if !seen.insert(chunk.id.as_str()) {
                return Err(Error::InvalidInput(format!(
                    "duplicate chunk id {:?} in source {source:?}",
                    chunk.id
                )));
            }
        }
        let tx = self.conn.transaction()?;
        delete_chunks(&tx, source)?;
        tx.execute(
            "INSERT INTO sources (id, content_hash) VALUES (?1, ?2) \
             ON CONFLICT(id) DO UPDATE SET content_hash = excluded.content_hash",
            params![source, content_hash],
        )?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO chunks (source, chunk_id, ordinal, title, heading, text, entities, metadata) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            let mut insert_fts = tx.prepare(
                "INSERT INTO chunks_fts (rowid, title, heading, text, entities) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for c in chunks {
                insert.execute(params![
                    source, c.id, c.ordinal, c.title, c.heading, c.text, c.entities, c.metadata
                ])?;
                let rowid = tx.last_insert_rowid();
                insert_fts.execute(params![rowid, c.title, c.heading, c.text, c.entities])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Remove a source and all its chunks. Returns whether it existed.
    pub fn delete_source(&mut self, source: &str) -> Result<bool> {
        let tx = self.conn.transaction()?;
        delete_chunks(&tx, source)?;
        let n = tx.execute("DELETE FROM sources WHERE id = ?1", [source])?;
        tx.commit()?;
        Ok(n > 0)
    }

    /// Every source with its hash, chunk count and embedded-chunk count, ordered by source.
    pub fn source_states(&self) -> Result<Vec<SourceState>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.content_hash, COUNT(c.id), COUNT(c.embedding) \
             FROM sources s LEFT JOIN chunks c ON c.source = s.id \
             GROUP BY s.id ORDER BY s.id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SourceState {
                source: r.get(0)?,
                content_hash: r.get(1)?,
                chunks: r.get::<_, i64>(2)?.unsigned_abs(),
                embedded_chunks: r.get::<_, i64>(3)?.unsigned_abs(),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Read a caller meta value (separate namespace from the index's own meta).
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM user_meta WHERE key = ?1", [key], |r| {
                r.get(0)
            })
            .optional()?)
    }

    /// Set (`Some`) or remove (`None`) a caller meta value.
    pub fn set_meta(&mut self, key: &str, value: Option<&str>) -> Result<()> {
        match value {
            Some(v) => self.conn.execute(
                "INSERT INTO user_meta (key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, v],
            )?,
            None => self
                .conn
                .execute("DELETE FROM user_meta WHERE key = ?1", [key])?,
        };
        Ok(())
    }
}

/// External-content FTS delete pattern: feed the old values back as a 'delete' row, then drop.
fn delete_chunks(tx: &rusqlite::Transaction<'_>, source: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO chunks_fts (chunks_fts, rowid, title, heading, text, entities) \
         SELECT 'delete', id, title, heading, text, entities FROM chunks WHERE source = ?1",
        [source],
    )?;
    tx.execute("DELETE FROM chunks WHERE source = ?1", [source])?;
    Ok(())
}
