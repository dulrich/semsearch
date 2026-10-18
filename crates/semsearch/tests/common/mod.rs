//! Test helpers: tmpfs scratch dirs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use semsearch::{Chunk, Index, IndexOptions};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique scratch dir under `/dev/shm` (when `TMPDIR` is unset) or `TMPDIR`; removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        let base = match std::env::var_os("TMPDIR") {
            None if Path::new("/dev/shm").is_dir() => PathBuf::from("/dev/shm"),
            _ => std::env::temp_dir(),
        };
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = base.join(format!(
            "semsearch-test-{}-{}-{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    pub fn db(&self) -> PathBuf {
        self.0.join("search.db")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn open(dir: &TempDir) -> Index {
    Index::open(dir.db(), &IndexOptions::default()).expect("open")
}

pub fn chunk(id: &str, title: &str, heading: &str, text: &str) -> Chunk {
    Chunk::new(id, title, heading, text)
}

pub fn sources(hits: &[semsearch::Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.source.as_str()).collect()
}
