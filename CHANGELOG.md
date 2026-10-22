# Changelog

All notable changes to `semsearch`. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning policy in `AGENTS.md`.

## [0.2.0] - 2026-10-22

### Changed (breaking)

- Corrupt derived on-disk data (bad `embedding_dim` meta, wrong-length embedding blob) is now `Error::Corrupt`, previously reported as `Error::InvalidInput`.
- `Error` is `#[non_exhaustive]`.

## [0.1.0] - 2026-10-17

### Added

- Initial release: hybrid retrieval over one derived SQLite file — FTS5 bm25, title-match, document-level term-coverage and caller-supplied vector legs, fused by reciprocal rank.
