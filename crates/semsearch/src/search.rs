//! Hybrid search: the companion's default `coverage` ranking mode (`runSearch`, search.mjs:465):
//! pooled bm25, title match, vector (when the query carries a vector and the index a space),
//! and term coverage, fused with RRF.

use std::collections::{HashMap, HashSet};

use rusqlite::{OptionalExtension, params};

use crate::error::{Error, Result};
use crate::fusion::{Leg, RRF_K, fuse};
use crate::query::{FtsQuery, build_fts_query, make_text_snippet, title_match_score};
use crate::store::Index;
use crate::vector::{dot, from_blob, normalized, sign_code};

/// Default and maximum result counts (search.mjs:28-29).
const DEFAULT_LIMIT: usize = 12;
const MAX_LIMIT: usize = 100;
/// Candidate pool = max(limit * FACTOR, MIN) (search.mjs:34-35).
const SEARCH_POOL_FACTOR: usize = 4;
const SEARCH_POOL_MIN: usize = 40;
/// Coverage needs at least this many terms, and a source must cover this many (ranking.mjs:34).
const COVERAGE_MIN_TERMS: usize = 2;
/// Per-leg RRF weights (ranking.mjs:17,27).
const TITLE_WEIGHT: f64 = 1.0;
const COVERAGE_WEIGHT: f64 = 1.0;
/// `RRF_VECTOR_WEIGHT` (ranking.mjs:18).
const RRF_VECTOR_WEIGHT: f64 = 1.0;

/// Phase 1: weighted bm25 (title 10, heading 5, text 1, entities 8), best chunk per source.
/// `MATERIALIZED` keeps bm25() out of the aggregate context.
const POOL_SQL: &str = "
WITH matched AS MATERIALIZED (
  SELECT rowid AS r, bm25(chunks_fts, 10.0, 5.0, 1.0, 8.0) AS score
  FROM chunks_fts WHERE chunks_fts MATCH ?1
)
SELECT c.id, c.source, MIN(m.score) AS score
FROM matched m JOIN chunks c ON c.id = m.r
GROUP BY c.source
ORDER BY score, c.source
LIMIT ?2";

/// Phase 2: display columns + FTS snippet for every pooled chunk, in one MATCH (a per-row
/// MATCH re-ran the full term match once per pooled row).
const HYDRATE_SQL: &str = "
SELECT c.id, c.chunk_id, c.title, c.heading, c.metadata,
       snippet(chunks_fts, 2, '', '', '...', 18)
FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?1 AND chunks_fts.rowid IN (SELECT value FROM json_each(?2))";

/// Coverage: presence only, one rowid-only statement per term; rowids resolve to sources
/// through the in-memory map (a JOIN to `chunks` measured ~7x slower on common terms).
const COVERAGE_SQL: &str = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1";

/// Leg-only rows: display columns for many chunks in one statement (no text; the snippet is
/// made only for rows that reach the final hit list).
const HYDRATE_CHUNKS_SQL: &str = "SELECT id, chunk_id, title, heading, metadata FROM chunks \
     WHERE id IN (SELECT value FROM json_each(?1))";
const CHUNK_TEXT_SQL: &str = "SELECT text FROM chunks WHERE id = ?1";

/// Vector rescore: the pooled candidates' f32 embeddings in one statement (sources resolve
/// through the in-memory rowid map).
const VECTOR_FETCH_SQL: &str =
    "SELECT id, embedding FROM vectors WHERE id IN (SELECT value FROM json_each(?1))";

/// A search query: text, plus an optional query embedding for the vector leg.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub text: String,
    /// Result count, clamped to 1..=100.
    pub limit: usize,
    /// Query embedding in the index's recorded space. `None` = keyword-only. A dim other than
    /// the space's is an error; an index with no space skips the vector leg.
    pub vector: Option<Vec<f32>>,
    /// Per-query override of [`crate::IndexOptions::vector_pool`].
    pub vector_pool: Option<usize>,
}

impl Query {
    /// Query with the default limit (12).
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            limit: DEFAULT_LIMIT,
            vector: None,
            vector_pool: None,
        }
    }

    /// Builder: attach a query embedding.
    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Builder: override the Hamming candidate pool.
    pub fn with_vector_pool(mut self, pool: usize) -> Self {
        self.vector_pool = Some(pool);
        self
    }

    /// Builder: set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// 1-based rank of a hit on each fusion leg; `None` = absent from that leg.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegRanks {
    pub text: Option<usize>,
    pub title: Option<usize>,
    pub vector: Option<usize>,
    pub coverage: Option<usize>,
}

/// One result: the best chunk of one source.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub source: String,
    pub chunk_id: String,
    pub title: String,
    pub heading: String,
    /// Fused RRF score, higher is better.
    pub score: f64,
    /// Negated weighted bm25 (higher is better); `None` for vector- or coverage-only hits.
    pub score_text: Option<f64>,
    pub snippet: String,
    pub metadata: Option<String>,
    pub ranks: LegRanks,
}

struct Row {
    /// Chunk rowid; only read for leg-only rows, whose snippet is made lazily.
    rowid: i64,
    source: String,
    chunk_id: String,
    title: String,
    heading: String,
    metadata: Option<String>,
    /// `None` until made (leg-only rows: from the chunk text, after fusion).
    snippet: Option<String>,
    score_text: Option<f64>,
}

impl Index {
    /// Hybrid search. Returns at most `query.limit` hits, one per source.
    pub fn search(&self, query: &Query) -> Result<Vec<Hit>> {
        let limit = query.limit.clamp(1, MAX_LIMIT);
        let query_vector = match (&query.vector, self.space()) {
            (Some(v), Some(space)) => Some(normalized(v, space.dim)?),
            _ => None,
        };
        let Some(fts) = build_fts_query(&query.text) else {
            return Ok(Vec::new());
        };
        let pool = (limit * SEARCH_POOL_FACTOR).max(SEARCH_POOL_MIN);
        let mut rows = self.pooled_search(&fts.primary, pool)?;
        let text_count = rows.len();
        let vector = match &query_vector {
            Some(v) => {
                let vpool = query.vector_pool.unwrap_or(self.vector_pool);
                self.vector_leg(v, vpool, &mut rows)?
            }
            None => None,
        };
        let coverage = self.coverage_leg(&fts, pool, &mut rows)?;

        let mut text_ranks: Vec<Option<usize>> = (1..=text_count).map(Some).collect();
        text_ranks.resize(rows.len(), None);
        let title_scores = rows
            .iter()
            .map(|r| {
                let s = title_match_score(&fts.terms, &r.title, &r.source);
                (s > 0.0).then_some(s)
            })
            .collect();
        let mut legs = vec![Leg {
            weight: TITLE_WEIGHT,
            scores: title_scores,
        }];
        let title_slot = Some(0);
        let mut vector_slot = None;
        let mut coverage_slot = None;
        if let Some(scores) = &vector {
            vector_slot = Some(legs.len());
            legs.push(Leg {
                weight: RRF_VECTOR_WEIGHT,
                scores: rows
                    .iter()
                    .map(|r| scores.get(&r.source).copied())
                    .collect(),
            });
        }
        if let Some(scores) = &coverage {
            coverage_slot = Some(legs.len());
            legs.push(Leg {
                weight: COVERAGE_WEIGHT,
                scores: rows
                    .iter()
                    .map(|r| scores.get(&r.source).copied())
                    .collect(),
            });
        }
        let fused = fuse(&text_ranks, &legs, RRF_K);
        let mut slots: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
        let mut hits = Vec::with_capacity(limit);
        for f in fused.into_iter().take(limit) {
            let Some(row) = slots.get_mut(f.index).and_then(Option::take) else {
                continue;
            };
            let rank = |slot: Option<usize>| slot.and_then(|s| f.leg_ranks[s]);
            hits.push(Hit {
                source: row.source,
                chunk_id: row.chunk_id,
                title: row.title,
                heading: row.heading,
                score: f.rrf,
                score_text: row.score_text,
                snippet: match row.snippet {
                    Some(s) => s,
                    None => self.text_snippet(row.rowid)?,
                },
                metadata: row.metadata,
                ranks: LegRanks {
                    text: text_ranks[f.index],
                    title: rank(title_slot),
                    vector: rank(vector_slot),
                    coverage: rank(coverage_slot),
                },
            });
        }
        Ok(hits)
    }

    /// Two-phase pooled bm25 (`runPooledSearch`, search.mjs:98).
    fn pooled_search(&self, matcher: &str, pool: usize) -> Result<Vec<Row>> {
        let mut stmt = self.conn.prepare_cached(POOL_SQL)?;
        let pooled: Vec<(i64, String, f64)> = stmt
            .query_map(params![matcher, pool as i64], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        if pooled.is_empty() {
            return Ok(Vec::new());
        }
        let ids = json_ids(pooled.iter().map(|p| p.0));
        let mut hydrate = self.conn.prepare_cached(HYDRATE_SQL)?;
        let mut extra: HashMap<i64, (String, String, String, Option<String>, String)> = hydrate
            .query_map(params![matcher, ids], |r| {
                Ok((
                    r.get(0)?,
                    (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?),
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut out = Vec::with_capacity(pooled.len());
        for (rowid, source, score) in pooled {
            let Some((chunk_id, title, heading, metadata, snippet)) = extra.remove(&rowid) else {
                continue;
            };
            out.push(Row {
                source,
                chunk_id,
                title,
                heading,
                metadata,
                rowid,
                snippet: Some(snippet),
                score_text: Some(-score),
            });
        }
        Ok(out)
    }

    /// Vector leg: Hamming pre-scan over the sign codes, exact f32 rescore of the pool, best
    /// chunk per source. Appends hydrated rows for sources not already in `rows`; returns
    /// per-source cosine scores, or `None` when nothing scored.
    fn vector_leg(
        &self,
        query: &[f32],
        pool: usize,
        rows: &mut Vec<Row>,
    ) -> Result<Option<HashMap<String, f64>>> {
        let best = self.vector_best(query, pool)?;
        if best.is_empty() {
            return Ok(None);
        }
        let known: HashSet<String> = rows.iter().map(|r| r.source.clone()).collect();
        // Deterministic append order: score desc, then source.
        let mut extra: Vec<(&String, &(f64, i64))> =
            best.iter().filter(|(s, _)| !known.contains(*s)).collect();
        extra.sort_by(|a, b| b.1.0.total_cmp(&a.1.0).then_with(|| a.0.cmp(b.0)));
        let wanted: Vec<(&str, i64)> = extra.iter().map(|(s, v)| (s.as_str(), v.1)).collect();
        rows.extend(self.hydrate_chunks(&wanted)?);
        Ok(Some(best.into_iter().map(|(s, (v, _))| (s, v)).collect()))
    }

    /// Pure vector retrieval (no text): the `k` nearest sources by best-chunk cosine, after
    /// the same Hamming pre-scan (`pool` candidates) and exact rescore the hybrid leg uses.
    /// Empty when the index has no space; a dim mismatch is [`Error::InvalidInput`].
    pub fn nearest_sources(
        &self,
        vector: &[f32],
        k: usize,
        pool: usize,
    ) -> Result<Vec<(String, f64)>> {
        let Some(space) = self.space() else {
            return Ok(Vec::new());
        };
        let q = normalized(vector, space.dim)?;
        let mut out: Vec<(String, f64)> = self
            .vector_best(&q, pool)?
            .into_iter()
            .map(|(s, (v, _))| (s, v))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(k);
        Ok(out)
    }

    /// Hamming pre-scan + exact rescore: per-source (best score, best chunk rowid).
    fn vector_best(&self, query: &[f32], pool: usize) -> Result<HashMap<String, (f64, i64)>> {
        let mut best: HashMap<String, (f64, i64)> = HashMap::new();
        let mut candidates = self.vectors.nearest(&sign_code(query), pool);
        candidates.sort_unstable();
        if candidates.is_empty() {
            return Ok(best);
        }
        let ids = json_ids(candidates.iter().copied());
        let mut stmt = self.conn.prepare_cached(VECTOR_FETCH_SQL)?;
        let mut q = stmt.query([ids])?;
        while let Some(r) = q.next()? {
            let rowid: i64 = r.get(0)?;
            let Some(source) = self.chunk_source.get(&rowid) else {
                continue;
            };
            let blob = r.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?;
            if blob.len() != query.len() * 4 {
                return Err(Error::InvalidInput(format!(
                    "chunk {rowid} embedding has {} bytes",
                    blob.len()
                )));
            }
            let score = f64::from(dot(query, &from_blob(blob)));
            let entry = best.entry(source.to_string()).or_insert((score, rowid));
            if score > entry.0 || (score == entry.0 && rowid < entry.1) {
                *entry = (score, rowid);
            }
        }
        drop(q);
        Ok(best)
    }

    /// Rows for chunks the FTS pool never returned, in input order: no bm25 score, snippet
    /// deferred to [`Index::text_snippet`].
    fn hydrate_chunks(&self, wanted: &[(&str, i64)]) -> Result<Vec<Row>> {
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare_cached(HYDRATE_CHUNKS_SQL)?;
        type Cols = (String, String, String, Option<String>);
        let mut found: HashMap<i64, Cols> = stmt
            .query_map([json_ids(wanted.iter().map(|w| w.1))], |r| {
                Ok((r.get(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(wanted
            .iter()
            .filter_map(|&(source, rowid)| {
                let (chunk_id, title, heading, metadata) = found.remove(&rowid)?;
                Some(Row {
                    rowid,
                    source: source.to_owned(),
                    chunk_id,
                    title,
                    heading,
                    metadata,
                    snippet: None,
                    score_text: None,
                })
            })
            .collect())
    }

    /// `make_text_snippet` of one chunk's text (the companion's leg-only snippet).
    fn text_snippet(&self, rowid: i64) -> Result<String> {
        let mut stmt = self.conn.prepare_cached(CHUNK_TEXT_SQL)?;
        let text: Option<String> = stmt.query_row([rowid], |r| r.get(0)).optional()?;
        Ok(text.as_deref().map(make_text_snippet).unwrap_or_default())
    }

    /// Document-level term coverage (`runCoverageLeg`, search.mjs:352). Appends hydrated rows
    /// for sources the FTS pool missed; returns per-source coverage scores, or `None` when the
    /// leg did not apply.
    fn coverage_leg(
        &self,
        fts: &FtsQuery,
        pool: usize,
        rows: &mut Vec<Row>,
    ) -> Result<Option<HashMap<String, f64>>> {
        let n_terms = fts.terms.len();
        if n_terms < COVERAGE_MIN_TERMS || fts.expanded.len() != n_terms {
            return Ok(None);
        }
        let mut stmt = self.conn.prepare_cached(COVERAGE_SQL)?;
        let mut source_terms: HashMap<String, usize> = HashMap::new();
        let mut chunk_terms: HashMap<i64, (usize, String)> = HashMap::new();
        for expression in &fts.expanded {
            let mut seen_sources = HashSet::new();
            let mut seen_chunks = HashSet::new();
            let mut q = stmt.query([expression])?;
            while let Some(r) = q.next()? {
                let rowid: i64 = r.get(0)?;
                let Some(source) = self.chunk_source.get(&rowid) else {
                    continue;
                };
                let source = source.to_string();
                if seen_chunks.insert(rowid) {
                    chunk_terms
                        .entry(rowid)
                        .or_insert_with(|| (0, source.clone()))
                        .0 += 1;
                }
                if seen_sources.insert(source.clone()) {
                    *source_terms.entry(source).or_insert(0) += 1;
                }
            }
        }
        let mut ranked: Vec<(String, f64)> = source_terms
            .into_iter()
            .filter(|(_, count)| *count >= COVERAGE_MIN_TERMS)
            .map(|(s, count)| (s, count as f64 / n_terms as f64))
            .collect();
        if ranked.is_empty() {
            return Ok(None);
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(pool);

        // Best-covering chunk per source (ties: lowest rowid, for determinism).
        let mut best: HashMap<&str, (usize, i64)> = HashMap::new();
        for (rowid, (count, source)) in &chunk_terms {
            let entry = best.entry(source.as_str()).or_insert((*count, *rowid));
            if *count > entry.0 || (*count == entry.0 && *rowid < entry.1) {
                *entry = (*count, *rowid);
            }
        }
        let known: HashSet<String> = rows.iter().map(|r| r.source.clone()).collect();
        let wanted: Vec<(&str, i64)> = ranked
            .iter()
            .filter(|(source, _)| !known.contains(source))
            .filter_map(|(source, _)| best.get(source.as_str()).map(|b| (source.as_str(), b.1)))
            .collect();
        rows.extend(self.hydrate_chunks(&wanted)?);
        Ok(Some(ranked.into_iter().collect()))
    }
}

/// A JSON array of rowids, for `json_each` binding.
fn json_ids(ids: impl Iterator<Item = i64>) -> String {
    let body: Vec<String> = ids.map(|i| i.to_string()).collect();
    format!("[{}]", body.join(","))
}
