//! Parity/latency harness (std + this crate's own deps only). Prints aggregate numbers only:
//! counts, recall, overlap ratios, latency percentiles, RSS. Never a path, title or text from
//! a real corpus; ranked paths are never printed in any mode.
//!
//! ```text
//! # real-corpus copy -> fresh scratch index (load phase), then a fresh process measures
//! bench load  --companion <sqlite> --index <scratch.db>
//! bench query --index <scratch.db> --queries <file> [--qvecs <le-f32 file>] [--reference <file>]
//!             [--pool N] [--rounds N] [--cleanup]
//! # no real corpus needed
//! bench synthetic --index <scratch.db> --n <sources> [--rounds N] [--cleanup]
//! ```
//!
//! `--qvecs`: one 1024-dim little-endian f32 vector per query line, concatenated.
//! `--reference`: one line per query, tab-separated source ids (Node companion top-10).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rusqlite::{Connection, OpenFlags};
use semsearch::{Chunk, Index, IndexOptions, Query, Space};

const DIM: usize = 1024;
const MODEL: &str = "bge-m3/f16";
const K: usize = 10;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: bench load --companion <sqlite> --index <scratch.db>
       bench query --index <scratch.db> --queries <file> [--qvecs <file>] [--reference <file>] [--pool N] [--rounds N] [--cleanup]
       bench synthetic --index <scratch.db> --n <sources> [--rounds N] [--cleanup]
       bench -h | --help

Parity/latency harness; prints aggregate numbers only (see the module docs).
";

fn main() -> Res<()> {
    if std::env::args().skip(1).any(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let mut flags: HashMap<String, String> = HashMap::new();
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        let key = rest[i].trim_start_matches("--").to_owned();
        if key == "cleanup" {
            flags.insert(key, String::new());
            i += 1;
        } else {
            let value = rest
                .get(i + 1)
                .cloned()
                .ok_or(format!("--{key} needs a value"))?;
            flags.insert(key, value);
            i += 2;
        }
    }
    let path = |k: &str| -> Res<PathBuf> {
        Ok(PathBuf::from(flags.get(k).ok_or(format!("missing --{k}"))?))
    };
    let num = |k: &str, d: usize| -> Res<usize> {
        Ok(flags.get(k).map(|v| v.parse()).transpose()?.unwrap_or(d))
    };
    let index_path = path("index")?;
    let result = match mode.as_str() {
        "load" => load(&path("companion")?, &index_path),
        "query" => query(
            &index_path,
            &path("queries")?,
            flags.get("qvecs").map(PathBuf::from),
            flags.get("reference").map(PathBuf::from),
            num("pool", semsearch::VECTOR_POOL)?,
            num("rounds", 3)?,
        ),
        "synthetic" => synthetic(&index_path, num("n", 2000)?, num("rounds", 3)?),
        _ => Err("mode: load | query | synthetic".into()),
    };
    if flags.contains_key("cleanup") {
        for suffix in ["", "-wal", "-shm"] {
            let mut p = index_path.clone().into_os_string();
            p.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(p));
        }
        println!("scratch index removed");
    }
    result
}

fn rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn load(companion: &Path, index_path: &Path) -> Res<()> {
    if index_path.exists() {
        return Err("scratch index already exists; pick a fresh path".into());
    }
    let src = Connection::open_with_flags(companion, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let space = Space::new(MODEL, DIM);
    let t0 = Instant::now();
    let mut index = Index::open(index_path, &IndexOptions::default())?;
    let mut stmt = src.prepare(
        "SELECT path, id, title, heading, text, entities, metadata_json, embedding, embedding_space \
         FROM chunks ORDER BY path, rowid",
    )?;
    let mut rows = stmt.query([])?;
    let (mut sources, mut chunks_n, mut embedded) = (0usize, 0usize, 0usize);
    let mut current: Option<String> = None;
    let mut batch: Vec<Chunk> = Vec::new();
    let flush = |index: &mut Index, source: &str, batch: &mut Vec<Chunk>| -> Res<()> {
        index.upsert_source_in(&space, source, "bench", batch)?;
        batch.clear();
        Ok(())
    };
    while let Some(r) = rows.next()? {
        let path: String = r.get(0)?;
        if current.as_deref() != Some(path.as_str()) {
            if let Some(prev) = current.take() {
                flush(&mut index, &prev, &mut batch)?;
                sources += 1;
            }
            current = Some(path);
        }
        let blob: Option<Vec<u8>> = r.get(7)?;
        let blob_space: Option<String> = r.get(8)?;
        let embedding = match (blob, blob_space.as_deref()) {
            (Some(b), Some(MODEL)) if b.len() == DIM * 4 => {
                embedded += 1;
                Some(
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect(),
                )
            }
            _ => None,
        };
        batch.push(Chunk {
            id: r.get(1)?,
            ordinal: batch.len() as i64,
            title: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            heading: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            text: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
            entities: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
            metadata: r.get(6)?,
            embedding,
        });
        chunks_n += 1;
    }
    if let Some(prev) = current.take() {
        flush(&mut index, &prev, &mut batch)?;
        sources += 1;
    }
    println!(
        "load: sources={sources} chunks={chunks_n} embedded={embedded} secs={:.1}",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn percentiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "n=0".into();
    }
    v.sort_by(f64::total_cmp);
    let p = |q: f64| v[((v.len() as f64 * q) as usize).min(v.len() - 1)];
    format!(
        "n={} p50={:.2}ms p90={:.2}ms max={:.2}ms",
        v.len(),
        p(0.5),
        p(0.9),
        v[v.len() - 1]
    )
}

fn timed_searches(index: &Index, queries: &[Query], rounds: usize) -> Res<Vec<f64>> {
    for q in queries {
        index.search(q)?; // warm-up
    }
    let mut out = Vec::new();
    for _ in 0..rounds {
        for q in queries {
            let t = Instant::now();
            std::hint::black_box(index.search(q)?);
            out.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(out)
}

/// Exact per-source best cosine over every stored vector, top `K` source ids.
fn exact_top(all: &[(String, Vec<f32>)], q: &[f32]) -> HashSet<String> {
    let mut best: HashMap<&str, f32> = HashMap::new();
    for (source, v) in all {
        let s: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
        let e = best.entry(source.as_str()).or_insert(f32::NEG_INFINITY);
        if s > *e {
            *e = s;
        }
    }
    let mut ranked: Vec<(&str, f32)> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .take(K)
        .map(|(s, _)| s.to_owned())
        .collect()
}

fn read_all_vectors(index_path: &Path) -> Res<Vec<(String, Vec<f32>)>> {
    let conn = Connection::open_with_flags(index_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt =
        conn.prepare("SELECT c.source, v.embedding FROM vectors v JOIN chunks c ON c.id = v.id")?;
    let rows = stmt.query_map([], |r| {
        let blob: Vec<u8> = r.get(1)?;
        Ok((
            r.get(0)?,
            blob.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn recall(index: &Index, index_path: &Path, qvecs: &[Vec<f32>], pool: usize) -> Res<()> {
    let all = read_all_vectors(index_path)?;
    let mut recalls = Vec::new();
    for q in qvecs {
        let truth = exact_top(&all, &unit(q));
        let got: HashSet<String> = index
            .nearest_sources(q, K, pool)?
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        recalls.push(truth.intersection(&got).count() as f64 / truth.len().max(1) as f64);
    }
    let mean = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
    let min = recalls.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "vector recall@{K} vs exact f32 (pool={pool}, vectors={}): mean={mean:.3} min={min:.2} n={}",
        all.len(),
        recalls.len()
    );
    Ok(())
}

fn query(
    index_path: &Path,
    queries_path: &Path,
    qvecs_path: Option<PathBuf>,
    reference: Option<PathBuf>,
    pool: usize,
    rounds: usize,
) -> Res<()> {
    let t0 = Instant::now();
    let opts = IndexOptions {
        vector_pool: pool,
        ..IndexOptions::default()
    };
    let index = Index::open_read_only(index_path, &opts)?;
    println!(
        "open: secs={:.2} rss_after_open={:.1}MiB embedded={}",
        t0.elapsed().as_secs_f64(),
        rss_mib(),
        index.embedded_count()
    );
    let texts: Vec<String> = std::fs::read_to_string(queries_path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect();
    let keyword: Vec<Query> = texts.iter().map(|t| Query::new(t).with_limit(K)).collect();
    println!(
        "keyword latency: {}",
        percentiles(timed_searches(&index, &keyword, rounds)?)
    );
    if let Some(reference) = reference {
        let refs: Vec<HashSet<String>> = std::fs::read_to_string(reference)?
            .lines()
            .map(|l| {
                l.split('\t')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .collect();
        let mut ratios = Vec::new();
        for (q, want) in keyword.iter().zip(&refs) {
            if want.is_empty() {
                continue;
            }
            let got: HashSet<String> = index.search(q)?.into_iter().map(|h| h.source).collect();
            ratios.push(got.intersection(want).count() as f64 / want.len() as f64);
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
        let min = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let exact = ratios.iter().filter(|r| **r >= 1.0).count();
        println!(
            "keyword top-{K} overlap vs reference: mean={mean:.3} min={min:.2} full={exact}/{} (refs={})",
            ratios.len(),
            refs.len()
        );
    }
    if let Some(qvecs_path) = qvecs_path {
        let bytes = std::fs::read(qvecs_path)?;
        let floats: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let qvecs: Vec<Vec<f32>> = floats
            .as_chunks::<DIM>()
            .0
            .iter()
            .map(|v| v.to_vec())
            .collect();
        if qvecs.len() != texts.len() {
            return Err(
                format!("{} query vectors for {} queries", qvecs.len(), texts.len()).into(),
            );
        }
        let hybrid: Vec<Query> = texts
            .iter()
            .zip(&qvecs)
            .map(|(t, v)| Query::new(t).with_limit(K).with_vector(v.clone()))
            .collect();
        println!(
            "hybrid latency: {}",
            percentiles(timed_searches(&index, &hybrid, rounds)?)
        );
        recall(&index, index_path, &qvecs, pool)?;
    }
    println!("rss_end={:.1}MiB", rss_mib());
    Ok(())
}

/// Random unit vectors + generated text; no real corpus.
fn synthetic(index_path: &Path, n: usize, rounds: usize) -> Res<()> {
    if index_path.exists() {
        return Err("scratch index already exists; pick a fresh path".into());
    }
    const WORDS: [&str; 16] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let space = Space::new("synthetic", DIM);
    let t0 = Instant::now();
    {
        let mut index = Index::open(index_path, &IndexOptions::default())?;
        for s in 0..n {
            let chunks: Vec<Chunk> = (0..4)
                .map(|c| {
                    let text: Vec<&str> = (0..40).map(|_| WORDS[(next() % 16) as usize]).collect();
                    let v: Vec<f32> = (0..DIM)
                        .map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                        .collect();
                    Chunk::new(&format!("c{c}"), &format!("doc {s}"), "", &text.join(" "))
                        .with_embedding(v)
                })
                .collect();
            index.upsert_source_in(&space, &format!("s{s}"), "h", &chunks)?;
        }
    }
    println!(
        "load: sources={n} chunks={} secs={:.1}",
        n * 4,
        t0.elapsed().as_secs_f64()
    );
    let index = Index::open_read_only(index_path, &IndexOptions::default())?;
    println!(
        "open: rss_after_open={:.1}MiB embedded={}",
        rss_mib(),
        index.embedded_count()
    );
    let mut qs: BTreeMap<usize, (String, Vec<f32>)> = BTreeMap::new();
    for i in 0..40 {
        let text = format!(
            "{} {}",
            WORDS[(next() % 16) as usize],
            WORDS[(next() % 16) as usize]
        );
        let v: Vec<f32> = (0..DIM)
            .map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect();
        qs.insert(i, (text, v));
    }
    let keyword: Vec<Query> = qs
        .values()
        .map(|(t, _)| Query::new(t).with_limit(K))
        .collect();
    let hybrid: Vec<Query> = qs
        .values()
        .map(|(t, v)| Query::new(t).with_limit(K).with_vector(v.clone()))
        .collect();
    println!(
        "keyword latency: {}",
        percentiles(timed_searches(&index, &keyword, rounds)?)
    );
    println!(
        "hybrid latency: {}",
        percentiles(timed_searches(&index, &hybrid, rounds)?)
    );
    let qvecs: Vec<Vec<f32>> = qs.into_values().map(|(_, v)| v).collect();
    recall(&index, index_path, &qvecs, semsearch::VECTOR_POOL)?;
    Ok(())
}
