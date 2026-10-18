use std::path::PathBuf;

/// Errors returned by [`crate::Index`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Underlying SQLite failure.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Read-only open of a file that does not exist.
    #[error("index not found: {0}")]
    NotFound(PathBuf),
    /// Read-only open of a file whose schema version is not [`crate::SCHEMA_VERSION`].
    #[error("schema version mismatch: found {found:?}, expected {expected}")]
    SchemaMismatch { found: Option<i64>, expected: i64 },
    /// Caller input rejected before reaching SQLite.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;
