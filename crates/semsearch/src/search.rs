//! Keyword search: the companion's default `coverage` ranking mode minus the vector leg
//! (`runSearch`, search.mjs:465).

use std::collections::{HashMap, HashSet};

use rusqlite::{OptionalExtension, params};

use crate::error::Result;
use crate::fusion::{Leg, RRF_K, fuse};
use crate::query::{FtsQuery, build_fts_query, make_text_snippet, title_match_score};
use crate::store::Index;

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

/// Phase 2: display columns + FTS snippet for one pooled chunk.
const HYDRATE_SQL: &str = "
SELECT c.chunk_id, c.title, c.heading, c.metadata,
       snippet(chunks_fts, 2, '', '', '...', 18)
FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?1 AND chunks_fts.rowid = ?2";

/// Coverage: presence only, one statement per term.
const COVERAGE_SQL: &str = "
SELECT c.id, c.source FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?1";

const HYDRATE_CHUNK_SQL: &str =
    "SELECT chunk_id, title, heading, text, metadata FROM chunks WHERE id = ?1";

/// A keyword query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub text: String,
    /// Result count, clamped to 1..=100.
    pub limit: usize,
}

impl Query {
    /// Query with the default limit (12).
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            limit: DEFAULT_LIMIT,
        }
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
    /// Negated weighted bm25 (higher is better); `None` for coverage-only hits.
    pub score_text: Option<f64>,
    pub snippet: String,
    pub metadata: Option<String>,
    pub ranks: LegRanks,
}

struct Row {
    source: String,
    chunk_id: String,
    title: String,
    heading: String,
    metadata: Option<String>,
    snippet: String,
    score_text: Option<f64>,
}

impl Index {
    /// Keyword search. Returns at most `query.limit` hits, one per source.
    pub fn search(&self, query: &Query) -> Result<Vec<Hit>> {
        let limit = query.limit.clamp(1, MAX_LIMIT);
        let Some(fts) = build_fts_query(&query.text) else {
            return Ok(Vec::new());
        };
        let pool = (limit * SEARCH_POOL_FACTOR).max(SEARCH_POOL_MIN);
        let mut rows = self.pooled_search(&fts.primary, pool)?;
        let text_count = rows.len();
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
        if let Some(scores) = &coverage {
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
            hits.push(Hit {
                source: row.source,
                chunk_id: row.chunk_id,
                title: row.title,
                heading: row.heading,
                score: f.rrf,
                score_text: row.score_text,
                snippet: row.snippet,
                metadata: row.metadata,
                ranks: LegRanks {
                    text: text_ranks[f.index],
                    title: f.leg_ranks.first().copied().flatten(),
                    coverage: f.leg_ranks.get(1).copied().flatten(),
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
        let mut hydrate = self.conn.prepare_cached(HYDRATE_SQL)?;
        let mut out = Vec::with_capacity(pooled.len());
        for (rowid, source, score) in pooled {
            let extra = hydrate
                .query_row(params![matcher, rowid], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })
                .optional()?;
            let Some((chunk_id, title, heading, metadata, snippet)) = extra else {
                continue;
            };
            out.push(Row {
                source,
                chunk_id,
                title,
                heading,
                metadata,
                snippet,
                score_text: Some(-score),
            });
        }
        Ok(out)
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
                let source: String = r.get(1)?;
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
        let mut hydrate = self.conn.prepare_cached(HYDRATE_CHUNK_SQL)?;
        for (source, _) in &ranked {
            if known.contains(source) {
                continue;
            }
            let Some(&(_, rowid)) = best.get(source.as_str()) else {
                continue;
            };
            let row = hydrate
                .query_row([rowid], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<String>>(4)?,
                    ))
                })
                .optional()?;
            let Some((chunk_id, title, heading, text, metadata)) = row else {
                continue;
            };
            rows.push(Row {
                source: source.clone(),
                chunk_id,
                title,
                heading,
                metadata,
                snippet: make_text_snippet(&text),
                score_text: None,
            });
        }
        Ok(Some(ranked.into_iter().collect()))
    }
}
