# Changelog

All notable changes to `semsearch`. Release rule: the `publish` skill's `## Releases` section; local specifics in `AGENTS.md` "Versioning".

## 0.3.1 — 2026-11-11

### Changed

- Index open computes sign codes straight from the stored blobs, without a per-vector `Vec<f32>` or `Vec<u64>`: allocation while opening a 95,019-vector index drops from 427.9 MiB to 45.2 MiB. No API, schema or result change.

## 0.3.0 — 2026-11-10

### Changed

- **Breaking:** `rusqlite` is now pinned `=0.41.1` (bundled SQLite 3.54.1 via `libsqlite3-sys` 0.39.1), fixing RUSTSEC-2026-0459 / CVE-2026-119375. `semsearch::Error::Sqlite` wraps `rusqlite::Error`, and `libsqlite3-sys` allows one version per build graph, so consumers must move to `rusqlite` 0.41 together.

## 0.2.1 — 2026-11-07

### Added

- `LICENSE` file (MIT).
- `authors` and `repository` package metadata.
- Public README.

### Changed

- The versioning policy now tags each release (`vX.Y.Z`).
- The bench example answers `--build` and `--version`.
- Rust toolchain pinned to 1.99.1.
- Tests write temporary databases on tmpfs.

The library API and index schema are unchanged.

## 0.2.0 — 2026-10-22

### Changed (breaking)

- Corrupt derived on-disk data (bad `embedding_dim` meta, wrong-length embedding blob) is now `Error::Corrupt`, previously reported as `Error::InvalidInput`.
- `Error` is `#[non_exhaustive]`.

## 0.1.0 — 2026-10-17

### Added

- Initial release: hybrid retrieval over one derived SQLite file — FTS5 bm25, title-match, document-level term-coverage and caller-supplied vector legs, fused by reciprocal rank.
