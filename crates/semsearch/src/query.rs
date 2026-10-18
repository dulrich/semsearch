//! Query tokenization and FTS5 expression building (port of the companion's `ranking.mjs`).

/// Cap on distinct query terms (`MAX_QUERY_TERMS`, ranking.mjs:11).
pub(crate) const MAX_QUERY_TERMS: usize = 24;
/// Minimum length (chars) of the trailing term for `"t"*` prefix expansion.
pub(crate) const PREFIX_MIN_CHARS: usize = 3;
/// Snippet token budget (`makeTextSnippet`, search.mjs:328).
pub(crate) const SNIPPET_TOKENS: usize = 18;

fn is_term_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '@' | '.' | '/' | ':' | '\'' | '-')
}

fn is_edge_punct(c: char) -> bool {
    matches!(c, '.' | '/' | ':' | '\'' | '-')
}

/// Lowercased, edge-trimmed, de-duplicated query terms, capped at [`MAX_QUERY_TERMS`].
pub(crate) fn tokenize_query(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut terms: Vec<String> = Vec::new();
    for raw in lower.split(|c: char| !is_term_char(c)) {
        let term = raw.trim_matches(is_edge_punct);
        if term.is_empty() || terms.iter().any(|t| t == term) {
            continue;
        }
        terms.push(term.to_owned());
        if terms.len() >= MAX_QUERY_TERMS {
            break;
        }
    }
    terms
}

pub(crate) fn quote_fts(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// The FTS5 expressions one query compiles to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FtsQuery {
    pub terms: Vec<String>,
    /// Per-term expressions; only the trailing term may carry `*`.
    pub expanded: Vec<String>,
    /// `(phrase) OR (t1 AND t2 AND ...)`.
    pub primary: String,
}

/// Build the primary FTS5 expression. Returns `None` when the query has no terms.
///
/// Changed from the companion: the trailing term is prefix-expanded only when it is at least
/// [`PREFIX_MIN_CHARS`] characters (matching the index's `prefix='3'`).
pub(crate) fn build_fts_query(query: &str) -> Option<FtsQuery> {
    let terms = tokenize_query(query);
    if terms.is_empty() {
        return None;
    }
    let mut expanded: Vec<String> = terms.iter().map(|t| quote_fts(t)).collect();
    if let (Some(last_term), Some(last)) = (terms.last(), expanded.last_mut())
        && last_term.chars().count() >= PREFIX_MIN_CHARS
    {
        last.push('*');
    }
    let phrase = quote_fts(&terms.join(" "));
    let primary = format!("({phrase}) OR ({})", expanded.join(" AND "));
    Some(FtsQuery {
        terms,
        expanded,
        primary,
    })
}

pub(crate) fn normalize_for_match(value: &str) -> String {
    let lower = value.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut pending_space = false;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        } else {
            pending_space = true;
        }
    }
    out
}

fn base_name(path: &str) -> &str {
    let tail = path.rsplit('/').next().unwrap_or(path);
    match tail.rfind('.') {
        Some(dot) if dot > 0 => &tail[..dot],
        _ => tail,
    }
}

/// How strongly a page's name answers the query, 0..=1 (`titleMatchScore`, ranking.mjs:153).
pub(crate) fn title_match_score(terms: &[String], title: &str, source: &str) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let phrase = normalize_for_match(&terms.join(" "));
    if phrase.is_empty() {
        return 0.0;
    }
    let title = normalize_for_match(title);
    let base = normalize_for_match(base_name(source));
    let path = normalize_for_match(source);
    if title == phrase || base == phrase {
        return 1.0;
    }
    if title.starts_with(&phrase) || base.starts_with(&phrase) {
        return 0.85;
    }
    if title.contains(&phrase) || base.contains(&phrase) {
        return 0.7;
    }
    let n = terms.len() as f64;
    let in_title = terms
        .iter()
        .filter(|t| title.contains(&normalize_for_match(t)))
        .count() as f64
        / n;
    let in_path = terms
        .iter()
        .filter(|t| path.contains(&normalize_for_match(t)))
        .count() as f64
        / n;
    let partial = in_title * 0.5 + in_path * 0.2;
    if partial > 0.0 {
        partial.min(0.65)
    } else {
        0.0
    }
}

/// Head of a chunk's text at [`SNIPPET_TOKENS`] (`makeTextSnippet`, search.mjs:328).
pub(crate) fn make_text_snippet(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let head = words
        .iter()
        .take(SNIPPET_TOKENS)
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    if words.len() > SNIPPET_TOKENS {
        format!("{head}...")
    } else {
        head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_trims_dedups_and_caps() {
        assert_eq!(
            tokenize_query("Hello, hello 'World'. a/b.md -x-"),
            vec!["hello", "world", "a/b.md", "x"]
        );
        let many: String = (0..40).map(|i| format!("t{i} ")).collect();
        assert_eq!(tokenize_query(&many).len(), MAX_QUERY_TERMS);
        assert!(tokenize_query("  ... --- ").is_empty());
    }

    #[test]
    fn prefix_only_on_last_token_of_three_or_more_chars() {
        let q = build_fts_query("alpha beta gam").expect("terms");
        assert_eq!(q.expanded, vec!["\"alpha\"", "\"beta\"", "\"gam\"*"]);
        assert_eq!(
            q.primary,
            "(\"alpha beta gam\") OR (\"alpha\" AND \"beta\" AND \"gam\"*)"
        );
        let short = build_fts_query("alpha ga").expect("terms");
        assert_eq!(short.expanded, vec!["\"alpha\"", "\"ga\""]);
        assert!(build_fts_query("!!!").is_none());
    }

    #[test]
    fn quotes_are_escaped() {
        assert_eq!(quote_fts("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn title_score_tiers() {
        let t = |q: &str| tokenize_query(q);
        assert_eq!(title_match_score(&t("rust book"), "Rust Book", "x.md"), 1.0);
        assert_eq!(title_match_score(&t("rust"), "Other", "dir/rust.md"), 1.0);
        assert_eq!(title_match_score(&t("rust"), "Rust Book", "x.md"), 0.85);
        assert_eq!(title_match_score(&t("book"), "Rust Book", "x.md"), 0.7);
        assert_eq!(title_match_score(&t("nothing"), "Rust Book", "x.md"), 0.0);
    }

    #[test]
    fn snippet_truncates() {
        let text: String = (0..20).map(|i| format!("w{i} ")).collect();
        let s = make_text_snippet(&text);
        assert!(s.ends_with("w17..."));
        assert_eq!(make_text_snippet("a  b"), "a b");
    }
}
