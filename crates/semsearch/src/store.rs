//! Storage: open/rebuild, schema, source writes, meta.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};

use crate::error::{Error, Result};
use crate::vector::{Codes, Space, VECTOR_POOL, from_blob, normalized, sign_code, to_blob};

/// Schema version stored in `meta`. A mismatch rebuilds an empty index; never migrates.
/// v2: embeddings moved from `chunks.embedding` to the `vectors` side table, so the rescore
/// fetch reads 4 KiB blobs without walking each chunk's text overflow pages.
pub const SCHEMA_VERSION: i64 = 2;

const META_SCHEMA_VERSION: &str = "schema_version";
/// Embedding-space identity (NULL until the first vector write).
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
  UNIQUE (source, chunk_id)
);
CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
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
    /// Hamming-scan candidates rescored exactly (default [`VECTOR_POOL`]); a
    /// [`crate::Query`] may override it per call.
    pub vector_pool: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            busy_timeout: Duration::from_secs(5),
            vector_pool: VECTOR_POOL,
        }
    }
}

/// One chunk of a source, as supplied by the caller.
#[derive(Debug, Clone, Default, PartialEq)]
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
    /// Optional embedding in the space passed to [`Index::upsert_source_in`]; stored
    /// L2-normalized as little-endian f32.
    pub embedding: Option<Vec<f32>>,
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

    /// Builder: attach an embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Per-source state for the caller's reconcile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceState {
    pub source: String,
    pub content_hash: String,
    pub chunks: u64,
    /// Chunks carrying an embedding.
    pub embedded_chunks: u64,
}

/// One search index over one SQLite file.
#[derive(Debug)]
pub struct Index {
    pub(crate) conn: Connection,
    rebuilt: bool,
    read_only: bool,
    space: Option<Space>,
    pub(crate) codes: Codes,
    pub(crate) vector_pool: usize,
    /// Chunk rowid -> source, for rowid-only coverage scans (the companion's coverage map).
    pub(crate) chunk_source: HashMap<i64, Arc<str>>,
}

impl Index {
    /// Open or create. On a missing schema or a version mismatch the file's tables are dropped
    /// and recreated empty, and [`Index::rebuilt`] returns `true`.
    pub fn open(path: impl AsRef<Path>, opts: &IndexOptions) -> Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        conn.busy_timeout(opts.busy_timeout)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let mut index = Self::with_conn(conn, false, opts);
        if index.schema_version()? != Some(SCHEMA_VERSION) {
            index.rebuild()?;
            index.rebuilt = true;
        }
        index.load_vectors()?;
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
        let mut index = Self::with_conn(conn, true, opts);
        let found = index.schema_version()?;
        if found != Some(SCHEMA_VERSION) {
            return Err(Error::SchemaMismatch {
                found,
                expected: SCHEMA_VERSION,
            });
        }
        index.load_vectors()?;
        Ok(index)
    }

    fn with_conn(conn: Connection, read_only: bool, opts: &IndexOptions) -> Self {
        Self {
            conn,
            rebuilt: false,
            read_only,
            space: None,
            codes: Codes::default(),
            vector_pool: opts.vector_pool,
            chunk_source: HashMap::new(),
        }
    }

    /// Build the in-memory rowid -> source map, then the recorded space and sign-code table.
    fn load_vectors(&mut self) -> Result<()> {
        let mut map = HashMap::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT id, source FROM chunks ORDER BY source")?;
            let mut rows = stmt.query([])?;
            let mut current: Option<Arc<str>> = None;
            while let Some(r) = rows.next()? {
                let rowid: i64 = r.get(0)?;
                let source = r.get_ref(1)?.as_str().map_err(rusqlite::Error::from)?;
                let shared = match &current {
                    Some(c) if &**c == source => c.clone(),
                    _ => {
                        let c: Arc<str> = Arc::from(source);
                        current = Some(c.clone());
                        c
                    }
                };
                map.insert(rowid, shared);
            }
        }
        self.chunk_source = map;
        let model = self.read_index_meta(META_EMBEDDING_MODEL)?;
        let dim = self.read_index_meta(META_EMBEDDING_DIM)?;
        self.space = match (model, dim) {
            (Some(model), Some(dim)) => Some(Space {
                model,
                dim: dim.parse().map_err(|_| {
                    Error::InvalidInput(format!("corrupt embedding_dim meta {dim:?}"))
                })?,
            }),
            _ => None,
        };
        let Some(space) = &self.space else {
            self.codes = Codes::default();
            return Ok(());
        };
        let mut codes = Codes::new(space.dim);
        let mut stmt = self
            .conn
            .prepare("SELECT id, embedding FROM vectors ORDER BY id")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let rowid: i64 = r.get(0)?;
            let blob = r.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?;
            if blob.len() != space.dim * 4 {
                return Err(Error::InvalidInput(format!(
                    "chunk {rowid} embedding has {} bytes, space dim {} needs {}",
                    blob.len(),
                    space.dim,
                    space.dim * 4
                )));
            }
            codes.insert(rowid, &sign_code(&from_blob(blob)));
        }
        drop(rows);
        drop(stmt);
        self.codes = codes;
        Ok(())
    }

    fn read_index_meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
                r.get::<_, Option<String>>(0)
            })
            .optional()?
            .flatten())
    }

    /// The recorded embedding space, if any vector was ever written (and not reset).
    pub fn space(&self) -> Option<&Space> {
        self.space.as_ref()
    }

    /// Number of chunks in the in-memory sign-code table (= embedded chunks).
    pub fn embedded_count(&self) -> usize {
        self.codes.len()
    }

    /// Deliberate space change: drop every stored embedding and the recorded space. Keyword
    /// data is kept; the caller re-embeds (in the new space) via [`Index::set_embeddings`].
    pub fn reset_space(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM vectors", [])?;
        tx.execute(
            "UPDATE meta SET value = NULL WHERE key IN (?1, ?2)",
            params![META_EMBEDDING_MODEL, META_EMBEDDING_DIM],
        )?;
        tx.commit()?;
        self.space = None;
        self.codes = Codes::default();
        Ok(())
    }

    /// Check `space` against the recorded one. `Ok(true)` = nothing recorded yet.
    fn check_space(&self, space: &Space) -> Result<bool> {
        if space.dim == 0 || space.model.is_empty() {
            return Err(Error::InvalidInput(
                "space needs a model and dim > 0".into(),
            ));
        }
        match &self.space {
            None => Ok(true),
            Some(recorded) if recorded == space => Ok(false),
            Some(recorded) => Err(Error::SpaceMismatch {
                recorded: recorded.clone(),
                got: space.clone(),
            }),
        }
    }

    fn record_space(tx: &rusqlite::Transaction<'_>, space: &Space) -> Result<()> {
        tx.execute(
            "UPDATE meta SET value = ?2 WHERE key = ?1",
            params![META_EMBEDDING_MODEL, space.model],
        )?;
        tx.execute(
            "UPDATE meta SET value = ?2 WHERE key = ?1",
            params![META_EMBEDDING_DIM, space.dim.to_string()],
        )?;
        Ok(())
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
        self.space = None;
        self.codes = Codes::default();
        self.chunk_source.clear();
        Ok(())
    }

    /// Atomically replace every chunk of `source` (creating the source if new). Keyword-only:
    /// a chunk carrying an embedding is rejected (use [`Index::upsert_source_in`]).
    pub fn upsert_source(
        &mut self,
        source: &str,
        content_hash: &str,
        chunks: &[Chunk],
    ) -> Result<()> {
        self.upsert_inner(None, source, content_hash, chunks)
    }

    /// [`Index::upsert_source`] with embeddings in `space`, written in the same transaction.
    /// The first vector write records the space; a different model or dim is
    /// [`Error::SpaceMismatch`] and nothing is written.
    pub fn upsert_source_in(
        &mut self,
        space: &Space,
        source: &str,
        content_hash: &str,
        chunks: &[Chunk],
    ) -> Result<()> {
        self.upsert_inner(Some(space), source, content_hash, chunks)
    }

    fn upsert_inner(
        &mut self,
        space: Option<&Space>,
        source: &str,
        content_hash: &str,
        chunks: &[Chunk],
    ) -> Result<()> {
        let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(chunks.len());
        let mut first_space = false;
        if let Some(space) = space {
            first_space = self.check_space(space)?;
        }
        for chunk in chunks {
            vectors.push(match (&chunk.embedding, space) {
                (None, _) => None,
                (Some(_), None) => {
                    return Err(Error::InvalidInput(format!(
                        "chunk {:?} carries an embedding but no space was given",
                        chunk.id
                    )));
                }
                (Some(v), Some(space)) => Some(normalized(v, space.dim)?),
            });
        }
        let any_vector = vectors.iter().any(Option::is_some);
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
        let removed = delete_chunks(&tx, source)?;
        tx.execute(
            "INSERT INTO sources (id, content_hash) VALUES (?1, ?2) \
             ON CONFLICT(id) DO UPDATE SET content_hash = excluded.content_hash",
            params![source, content_hash],
        )?;
        let mut added: Vec<(i64, Vec<u64>)> = Vec::new();
        let mut rowids: Vec<i64> = Vec::with_capacity(chunks.len());
        {
            let mut insert = tx.prepare(
                "INSERT INTO chunks (source, chunk_id, ordinal, title, heading, text, entities, metadata) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            let mut insert_vector =
                tx.prepare("INSERT INTO vectors (id, embedding) VALUES (?1, ?2)")?;
            let mut insert_fts = tx.prepare(
                "INSERT INTO chunks_fts (rowid, title, heading, text, entities) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (c, v) in chunks.iter().zip(&vectors) {
                insert.execute(params![
                    source, c.id, c.ordinal, c.title, c.heading, c.text, c.entities, c.metadata
                ])?;
                let rowid = tx.last_insert_rowid();
                insert_fts.execute(params![rowid, c.title, c.heading, c.text, c.entities])?;
                rowids.push(rowid);
                if let Some(v) = v {
                    insert_vector.execute(params![rowid, to_blob(v)])?;
                    added.push((rowid, sign_code(v)));
                }
            }
        }
        let recorded = match space {
            Some(space) if first_space && any_vector => {
                Self::record_space(&tx, space)?;
                Some(space.clone())
            }
            _ => None,
        };
        tx.commit()?;
        if let Some(space) = recorded {
            self.codes = Codes::new(space.dim);
            self.space = Some(space);
        }
        for rowid in removed {
            self.codes.remove(rowid);
            self.chunk_source.remove(&rowid);
        }
        let shared: Arc<str> = Arc::from(source);
        for rowid in rowids {
            self.chunk_source.insert(rowid, shared.clone());
        }
        for (rowid, code) in added {
            self.codes.insert(rowid, &code);
        }
        Ok(())
    }

    /// Backfill embeddings of an already-indexed source, by chunk id. An unknown source or
    /// chunk is [`Error::InvalidInput`]; a space mismatch is [`Error::SpaceMismatch`]. Either
    /// way nothing is written.
    pub fn set_embeddings(
        &mut self,
        space: &Space,
        source: &str,
        embeddings: &[(String, Vec<f32>)],
    ) -> Result<()> {
        let first_space = self.check_space(space)?;
        let vectors: Vec<(&str, Vec<f32>)> = embeddings
            .iter()
            .map(|(id, v)| Ok((id.as_str(), normalized(v, space.dim)?)))
            .collect::<Result<_>>()?;
        let tx = self.conn.transaction()?;
        let known: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sources WHERE id = ?1)",
            [source],
            |r| r.get(0),
        )?;
        if !known {
            return Err(Error::InvalidInput(format!("unknown source {source:?}")));
        }
        let mut added = Vec::with_capacity(vectors.len());
        {
            let mut lookup =
                tx.prepare("SELECT id FROM chunks WHERE source = ?1 AND chunk_id = ?2")?;
            let mut upsert = tx.prepare(
                "INSERT INTO vectors (id, embedding) VALUES (?1, ?2) \
                 ON CONFLICT(id) DO UPDATE SET embedding = excluded.embedding",
            )?;
            for (id, v) in &vectors {
                let rowid: Option<i64> = lookup
                    .query_row(params![source, id], |r| r.get(0))
                    .optional()?;
                let Some(rowid) = rowid else {
                    return Err(Error::InvalidInput(format!(
                        "unknown chunk {id:?} in source {source:?}"
                    )));
                };
                upsert.execute(params![rowid, to_blob(v)])?;
                added.push((rowid, sign_code(v)));
            }
        }
        let record = first_space && !vectors.is_empty();
        if record {
            Self::record_space(&tx, space)?;
        }
        tx.commit()?;
        if record {
            self.codes = Codes::new(space.dim);
            self.space = Some(space.clone());
        }
        for (rowid, code) in added {
            self.codes.insert(rowid, &code);
        }
        Ok(())
    }

    /// Remove a source and all its chunks. Returns whether it existed.
    pub fn delete_source(&mut self, source: &str) -> Result<bool> {
        let tx = self.conn.transaction()?;
        let removed = delete_chunks(&tx, source)?;
        let n = tx.execute("DELETE FROM sources WHERE id = ?1", [source])?;
        tx.commit()?;
        for rowid in removed {
            self.codes.remove(rowid);
            self.chunk_source.remove(&rowid);
        }
        Ok(n > 0)
    }

    /// Every source with its hash, chunk count and embedded-chunk count, ordered by source.
    pub fn source_states(&self) -> Result<Vec<SourceState>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.content_hash, COUNT(c.id), COUNT(v.id) \
             FROM sources s LEFT JOIN chunks c ON c.source = s.id \
             LEFT JOIN vectors v ON v.id = c.id \
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
/// Returns the rowids of the removed chunks.
fn delete_chunks(tx: &rusqlite::Transaction<'_>, source: &str) -> Result<Vec<i64>> {
    let removed: Vec<i64> = {
        let mut stmt = tx.prepare("SELECT id FROM chunks WHERE source = ?1")?;
        stmt.query_map([source], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    tx.execute(
        "INSERT INTO chunks_fts (chunks_fts, rowid, title, heading, text, entities) \
         SELECT 'delete', id, title, heading, text, entities FROM chunks WHERE source = ?1",
        [source],
    )?;
    tx.execute(
        "DELETE FROM vectors WHERE id IN (SELECT id FROM chunks WHERE source = ?1)",
        [source],
    )?;
    tx.execute("DELETE FROM chunks WHERE source = ?1", [source])?;
    Ok(removed)
}
