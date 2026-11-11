# Changelog

All notable changes to `semsearch`. Release rule: the `publish` skill's `## Releases` section; local specifics in `AGENTS.md` "Versioning".

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
