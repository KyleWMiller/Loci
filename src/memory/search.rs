use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::HashMap;

use crate::memory::types::{MemoryType, Scope};

// ── Public types ──────────────────────────────────────────────────────────────

/// A single search result with full content.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub content: String,
    pub confidence: f64,
    pub score: f64,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// A compact summary result for progressive disclosure.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryResult {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub preview: String,
    pub score: f64,
}

/// Response from recall_by_query or recall_by_ids.
#[derive(Debug, Serialize)]
pub struct RecallResponse {
    pub results: Vec<SearchResult>,
    pub total_matched: usize,
    pub token_estimate: usize,
}

/// Response with summary-only results.
#[derive(Debug, Serialize)]
pub struct RecallSummaryResponse {
    pub results: Vec<SummaryResult>,
    pub total_matched: usize,
    pub token_estimate: usize,
}

/// Filters applied after RRF merge.
pub struct SearchFilter {
    pub memory_type: Option<MemoryType>,
    pub scope: Option<Scope>,
    pub group: String,
    pub min_confidence: f64,
}

/// Search configuration knobs.
pub struct SearchConfig {
    pub max_results: usize,
    pub token_budget: usize,
    pub rrf_k: usize,
}

/// Full inspection response for a single memory.
#[derive(Debug, Serialize)]
pub struct InspectResponse {
    pub memory: InspectMemory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations: Option<Vec<RelationEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<Vec<LogEntry>>,
}

#[derive(Debug, Serialize)]
pub struct InspectMemory {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub content: String,
    pub confidence: f64,
    pub access_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct RelationEntry {
    pub predicate: String,
    pub object: RelationTarget,
}

#[derive(Debug, Serialize)]
pub struct RelationTarget {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub preview: String,
}

#[derive(Debug, Serialize)]
pub struct LogEntry {
    pub operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub created_at: String,
}

// ── Internal row struct for fetched memories ──────────────────────────────────

struct MemoryRow {
    id: String,
    memory_type: String,
    content: String,
    source_group: Option<String>,
    scope: String,
    confidence: f64,
    access_count: u32,
    superseded_by: Option<String>,
    created_at: String,
    metadata: Option<serde_json::Value>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Hybrid search: vector KNN + FTS5 BM25 → RRF merge → filter → budget → track.
pub fn recall_by_query(
    conn: &Connection,
    query_embedding: &[f32],
    query_text: &str,
    filter: &SearchFilter,
    config: &SearchConfig,
) -> Result<RecallResponse> {
    let candidate_limit = config.max_results * 3;

    // 1. Vector KNN search
    let vec_results = vector_search(conn, query_embedding, candidate_limit)?;

    // 2. FTS5 BM25 search
    let fts_results = fts_search(conn, query_text, candidate_limit)?;

    // 3. RRF merge
    let merged = rrf_merge(&vec_results, &fts_results, config.rrf_k);

    // 4. Fetch full records for all candidate IDs
    let candidate_ids: Vec<&str> = merged.iter().map(|(id, _)| id.as_str()).collect();
    let memories = fetch_memories(conn, &candidate_ids)?;

    // 5. Post-filter and build ordered results
    let mut filtered: Vec<(MemoryRow, f64)> = Vec::new();
    for (id, score) in &merged {
        if let Some(mem) = memories.get(id.as_str()) {
            // Skip superseded
            if mem.superseded_by.is_some() {
                continue;
            }
            // Scope filter: always include global; include group only if matching
            match mem.scope.as_str() {
                "global" => {}
                "group" => {
                    if mem.source_group.as_deref() != Some(filter.group.as_str()) {
                        continue;
                    }
                }
                _ => continue,
            }
            // If caller specified scope filter, enforce it
            if let Some(ref scope_filter) = filter.scope {
                if mem.scope != scope_filter.as_str() {
                    continue;
                }
            }
            // Type filter
            if let Some(ref type_filter) = filter.memory_type {
                if mem.memory_type != type_filter.as_str() {
                    continue;
                }
            }
            // Confidence floor
            if mem.confidence < filter.min_confidence {
                continue;
            }
            filtered.push((
                MemoryRow {
                    id: mem.id.clone(),
                    memory_type: mem.memory_type.clone(),
                    content: mem.content.clone(),
                    source_group: mem.source_group.clone(),
                    scope: mem.scope.clone(),
                    confidence: mem.confidence,
                    access_count: mem.access_count,
                    superseded_by: mem.superseded_by.clone(),
                    created_at: mem.created_at.clone(),
                    metadata: mem.metadata.clone(),
                },
                *score,
            ));
        }
    }

    let total_matched = filtered.len();

    // 6. Token budget enforcement
    let mut token_sum = 0usize;
    let mut budgeted: Vec<(MemoryRow, f64)> = Vec::new();
    for (mem, score) in filtered {
        let tokens = mem.content.len() / 4;
        if !budgeted.is_empty() && token_sum + tokens > config.token_budget {
            break;
        }
        token_sum += tokens;
        budgeted.push((mem, score));
        if budgeted.len() >= config.max_results {
            break;
        }
    }

    // 7. Access tracking
    let returned_ids: Vec<&str> = budgeted.iter().map(|(m, _)| m.id.as_str()).collect();
    update_access(conn, &returned_ids)?;

    // 8. Build response
    let results: Vec<SearchResult> = budgeted
        .into_iter()
        .map(|(mem, score)| SearchResult {
            id: mem.id,
            memory_type: mem.memory_type,
            content: mem.content,
            confidence: mem.confidence,
            score,
            created_at: mem.created_at,
            metadata: mem.metadata,
        })
        .collect();

    Ok(RecallResponse {
        results,
        total_matched,
        token_estimate: token_sum,
    })
}

/// Direct hydration by IDs — no search, no filtering.
pub fn recall_by_ids(conn: &Connection, ids: &[String]) -> Result<RecallResponse> {
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let memories = fetch_memories(conn, &id_refs)?;

    let mut results: Vec<SearchResult> = Vec::new();
    let mut token_sum = 0usize;

    // Preserve input order
    for id in ids {
        if let Some(mem) = memories.get(id.as_str()) {
            token_sum += mem.content.len() / 4;
            results.push(SearchResult {
                id: mem.id.clone(),
                memory_type: mem.memory_type.clone(),
                content: mem.content.clone(),
                confidence: mem.confidence,
                score: 1.0, // No search score for direct hydration
                created_at: mem.created_at.clone(),
                metadata: mem.metadata.clone(),
            });
        }
    }

    let total = results.len();
    update_access(conn, &id_refs)?;

    Ok(RecallResponse {
        results,
        total_matched: total,
        token_estimate: token_sum,
    })
}

/// Convert full results to summary format.
pub fn to_summary(response: &RecallResponse) -> RecallSummaryResponse {
    let results: Vec<SummaryResult> = response
        .results
        .iter()
        .map(|r| SummaryResult {
            id: r.id.clone(),
            memory_type: r.memory_type.clone(),
            preview: truncate_preview(&r.content, 80),
            score: r.score,
        })
        .collect();

    let token_estimate = results
        .iter()
        .map(|r| r.preview.len() / 4 + 10) // preview + id/type/score overhead
        .sum();

    RecallSummaryResponse {
        results,
        total_matched: response.total_matched,
        token_estimate,
    }
}

/// Inspect a single memory by ID with optional relations and audit log.
pub fn inspect_memory(
    conn: &Connection,
    memory_id: &str,
    include_relations: bool,
    include_log: bool,
) -> Result<InspectResponse> {
    // Fetch the memory
    let memory = conn
        .query_row(
            "SELECT id, type, content, source_group, scope, confidence, access_count, \
             last_accessed, created_at, updated_at, superseded_by, metadata \
             FROM memories WHERE id = ?1",
            params![memory_id],
            |row| {
                let metadata_str: Option<String> = row.get(11)?;
                Ok(InspectMemory {
                    id: row.get(0)?,
                    memory_type: row.get(1)?,
                    content: row.get(2)?,
                    confidence: row.get(5)?,
                    access_count: row.get(6)?,
                    last_accessed: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    metadata: metadata_str
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                anyhow::anyhow!("memory not found: {memory_id}")
            }
            other => anyhow::anyhow!("database error: {other}"),
        })?;

    // Fetch relations
    let relations = if include_relations {
        let mut stmt = conn.prepare(
            "SELECT er.predicate, m.id, m.type, m.content \
             FROM entity_relations er \
             JOIN memories m ON er.object_id = m.id \
             WHERE er.subject_id = ?1",
        )?;
        let rows = stmt
            .query_map(params![memory_id], |row| {
                let content: String = row.get(3)?;
                Ok(RelationEntry {
                    predicate: row.get(0)?,
                    object: RelationTarget {
                        id: row.get(1)?,
                        memory_type: row.get(2)?,
                        preview: truncate_preview(&content, 100),
                    },
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Some(rows)
    } else {
        None
    };

    // Fetch audit log
    let log = if include_log {
        let mut stmt = conn.prepare(
            "SELECT operation, details, created_at \
             FROM memory_log WHERE memory_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt
            .query_map(params![memory_id], |row| {
                let details_str: Option<String> = row.get(1)?;
                Ok(LogEntry {
                    operation: row.get(0)?,
                    details: details_str
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    created_at: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Some(rows)
    } else {
        None
    };

    Ok(InspectResponse {
        memory,
        relations,
        log,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Vector KNN search via sqlite-vec.
fn vector_search(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<(String, f64)>> {
    let embedding_bytes = super::embedding_to_bytes(embedding);
    let mut stmt = conn.prepare(
        "SELECT id, distance FROM memories_vec \
         WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
    )?;
    let results = stmt
        .query_map(params![embedding_bytes, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

/// FTS5 BM25 keyword search.
///
/// Returns (id, rank) pairs. FTS5 rank is negative (more negative = better),
/// so we negate it for consistent ordering.
fn fts_search(conn: &Connection, query_text: &str, limit: usize) -> Result<Vec<(String, f64)>> {
    // Escape the query for FTS5: wrap each word in double quotes to avoid syntax errors
    let escaped = escape_fts_query(query_text);
    if escaped.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn.prepare(
        "SELECT id, rank FROM memories_fts \
         WHERE memories_fts MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let results = stmt
        .query_map(params![escaped, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

/// Escape a user query for FTS5 MATCH syntax.
///
/// Wraps each whitespace-delimited word in double quotes and joins with spaces
/// so FTS5 treats them as individual terms (implicit AND). Strips empty tokens.
fn escape_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|word| {
            // Strip any existing quotes and wrap in fresh ones
            let clean = word.replace('"', "");
            format!("\"{clean}\"")
        })
        .filter(|w| w != "\"\"")
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reciprocal Rank Fusion merge.
///
/// Combines ranked lists from vector and FTS search. Documents appearing in
/// both lists get additive scores; those in only one list get a single score.
fn rrf_merge(
    vec_results: &[(String, f64)],
    fts_results: &[(String, f64)],
    k: usize,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (id, _distance)) in vec_results.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k as f64 + rank as f64);
    }

    for (rank, (id, _rank_score)) in fts_results.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k as f64 + rank as f64);
    }

    let mut merged: Vec<(String, f64)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Batch-fetch memory records by IDs.
fn fetch_memories(conn: &Connection, ids: &[&str]) -> Result<HashMap<String, MemoryRow>> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }

    // Build a parameterized IN clause
    let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
    let sql = format!(
        "SELECT id, type, content, source_group, scope, confidence, access_count, \
         superseded_by, created_at, metadata \
         FROM memories WHERE id IN ({})",
        placeholders.join(", ")
    );

    let mut stmt = conn.prepare(&sql)?;

    let params: Vec<&dyn rusqlite::types::ToSql> =
        ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    let rows = stmt
        .query_map(params.as_slice(), |row| {
            let metadata_str: Option<String> = row.get(9)?;
            Ok(MemoryRow {
                id: row.get(0)?,
                memory_type: row.get(1)?,
                content: row.get(2)?,
                source_group: row.get(3)?,
                scope: row.get(4)?,
                confidence: row.get(5)?,
                access_count: row.get(6)?,
                superseded_by: row.get(7)?,
                created_at: row.get(8)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut map = HashMap::new();
    for row in rows {
        map.insert(row.id.clone(), row);
    }
    Ok(map)
}

/// Batch update access_count and last_accessed for returned results.
fn update_access(conn: &Connection, ids: &[&str]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let now = chrono::Utc::now().to_rfc3339();
    let mut stmt = conn.prepare(
        "UPDATE memories SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
    )?;
    for id in ids {
        stmt.execute(params![now, id])?;
    }
    Ok(())
}

/// Truncate content to max_chars, appending "..." if truncated.
fn truncate_preview(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        // Find a clean char boundary
        let end = content
            .char_indices()
            .take_while(|(i, _)| *i < max_chars)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(max_chars);
        format!("{}...", &content[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::store;

    fn test_db() -> Connection {
        db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::db::schema::init_schema(&conn).unwrap();
        conn
    }

    /// Unit vector along dimension 0.
    fn embedding_a() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[0] = 1.0;
        v
    }

    /// Unit vector along dimension 100 — orthogonal to embedding_a.
    fn embedding_b() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[100] = 1.0;
        v
    }

    /// Helper: insert a test memory directly via the store module.
    fn insert_test_memory(
        conn: &mut Connection,
        content: &str,
        memory_type: MemoryType,
        scope: Scope,
        group: &str,
        confidence: f64,
        embedding: &[f32],
    ) -> String {
        store::store_memory(
            conn,
            content,
            memory_type,
            scope,
            Some(group),
            confidence,
            None,
            None,
            embedding,
            0.92,
        )
        .unwrap()
        .id
    }

    fn default_filter(group: &str) -> SearchFilter {
        SearchFilter {
            memory_type: None,
            scope: None,
            group: group.to_string(),
            min_confidence: 0.1,
        }
    }

    fn default_config() -> SearchConfig {
        SearchConfig {
            max_results: 5,
            token_budget: 4000,
            rrf_k: 60,
        }
    }

    #[test]
    fn test_vector_search_returns_nearest() {
        let mut conn = test_db();
        let id_a = insert_test_memory(
            &mut conn,
            "Alpha memory about Rust",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        let _id_b = insert_test_memory(
            &mut conn,
            "Beta memory about Python",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_b(),
        );

        // Search with embedding_a — should find alpha first
        let results = vector_search(&conn, &embedding_a(), 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id_a);
        assert!(results[0].1 < 0.01); // very close distance
    }

    #[test]
    fn test_fts_search_matches_keywords() {
        let mut conn = test_db();
        let id_a = insert_test_memory(
            &mut conn,
            "The quantum computer operates at very low temperatures",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        let _id_b = insert_test_memory(
            &mut conn,
            "Rust is a systems programming language",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_b(),
        );

        let results = fts_search(&conn, "quantum computer", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn test_rrf_merge_combines_signals() {
        let vec_results = vec![
            ("doc_a".to_string(), 0.1),
            ("doc_b".to_string(), 0.3),
            ("doc_c".to_string(), 0.5),
        ];
        let fts_results = vec![
            ("doc_b".to_string(), -5.0),
            ("doc_a".to_string(), -3.0),
            ("doc_d".to_string(), -1.0),
        ];

        let merged = rrf_merge(&vec_results, &fts_results, 60);

        // doc_a and doc_b appear in both lists, should score higher
        let scores: HashMap<String, f64> = merged.into_iter().collect();
        assert!(scores["doc_a"] > scores["doc_c"]); // doc_a in both, doc_c in one
        assert!(scores["doc_b"] > scores["doc_d"]); // doc_b in both, doc_d in one
    }

    #[test]
    fn test_post_filter_excludes_superseded() {
        let mut conn = test_db();
        let id_old = insert_test_memory(
            &mut conn,
            "Old fact about Rust",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        // Supersede the old one
        let _id_new = store::store_memory(
            &mut conn,
            "Updated fact about Rust",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            Some(&id_old),
            &embedding_b(),
            0.92,
        )
        .unwrap()
        .id;

        // Search with embedding_a — should NOT return the superseded old memory
        let response = recall_by_query(
            &conn,
            &embedding_a(),
            "fact about Rust",
            &default_filter("default"),
            &default_config(),
        )
        .unwrap();

        let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&id_old.as_str()));
    }

    #[test]
    fn test_post_filter_by_type() {
        let mut conn = test_db();
        let id_sem = insert_test_memory(
            &mut conn,
            "Semantic knowledge about databases",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        let id_epi = insert_test_memory(
            &mut conn,
            "Episodic event about databases",
            MemoryType::Episodic,
            Scope::Group,
            "default",
            1.0,
            &embedding_b(),
        );

        let filter = SearchFilter {
            memory_type: Some(MemoryType::Semantic),
            scope: None,
            group: "default".to_string(),
            min_confidence: 0.1,
        };

        let response =
            recall_by_query(&conn, &embedding_a(), "databases", &filter, &default_config())
                .unwrap();

        let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&id_sem.as_str()));
        assert!(!ids.contains(&id_epi.as_str()));
    }

    #[test]
    fn test_post_filter_by_scope() {
        let mut conn = test_db();
        let id_global = insert_test_memory(
            &mut conn,
            "Global knowledge for everyone",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        let id_group = insert_test_memory(
            &mut conn,
            "Group-specific event log",
            MemoryType::Episodic,
            Scope::Group,
            "project-x",
            1.0,
            &embedding_b(),
        );

        // Search from "default" group — should see global but NOT project-x group memory
        let response = recall_by_query(
            &conn,
            &embedding_a(),
            "knowledge event",
            &default_filter("default"),
            &default_config(),
        )
        .unwrap();

        let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&id_global.as_str()));
        assert!(!ids.contains(&id_group.as_str()));
    }

    #[test]
    fn test_confidence_floor() {
        let mut conn = test_db();
        let _id_high = insert_test_memory(
            &mut conn,
            "High confidence fact",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            0.9,
            &embedding_a(),
        );
        let id_low = insert_test_memory(
            &mut conn,
            "Low confidence guess",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            0.05,
            &embedding_b(),
        );

        let response = recall_by_query(
            &conn,
            &embedding_a(),
            "fact guess",
            &default_filter("default"),
            &default_config(),
        )
        .unwrap();

        let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&id_low.as_str()));
    }

    #[test]
    fn test_token_budget_truncates() {
        let mut conn = test_db();
        // Each memory is ~100 chars = ~25 tokens
        for i in 0..10 {
            let mut emb = vec![0.0f32; 384];
            emb[i] = 1.0;
            insert_test_memory(
                &mut conn,
                &format!("Memory number {i} with enough content to take up some token budget space in the response payload"),
                MemoryType::Semantic,
                Scope::Global,
                "default",
                1.0,
                &emb,
            );
        }

        let config = SearchConfig {
            max_results: 10,
            token_budget: 50, // Very tight budget — ~200 chars
            rrf_k: 60,
        };

        let response = recall_by_query(
            &conn,
            &embedding_a(),
            "memory content",
            &default_filter("default"),
            &config,
        )
        .unwrap();

        // Should have fewer results than total due to budget
        assert!(response.results.len() < 10);
        assert!(response.token_estimate <= 75); // some slack
    }

    #[test]
    fn test_summary_only_mode() {
        let response = RecallResponse {
            results: vec![SearchResult {
                id: "test-id".to_string(),
                memory_type: "semantic".to_string(),
                content: "This is a fairly long piece of content that should be truncated to eighty characters when shown in summary mode for progressive disclosure".to_string(),
                confidence: 0.9,
                score: 0.03,
                created_at: "2026-01-01T00:00:00Z".to_string(),
                metadata: None,
            }],
            total_matched: 1,
            token_estimate: 35,
        };

        let summary = to_summary(&response);
        assert_eq!(summary.results.len(), 1);
        assert!(summary.results[0].preview.len() <= 83); // 80 + "..."
        assert!(summary.results[0].preview.ends_with("..."));
    }

    #[test]
    fn test_recall_by_ids() {
        let mut conn = test_db();
        let id_a = insert_test_memory(
            &mut conn,
            "Memory alpha",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );
        let id_b = insert_test_memory(
            &mut conn,
            "Memory beta",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_b(),
        );

        let response =
            recall_by_ids(&conn, &[id_b.clone(), id_a.clone()]).unwrap();

        assert_eq!(response.results.len(), 2);
        // Order should match input
        assert_eq!(response.results[0].id, id_b);
        assert_eq!(response.results[1].id, id_a);
    }

    #[test]
    fn test_access_tracking() {
        let mut conn = test_db();
        let id = insert_test_memory(
            &mut conn,
            "Trackable memory",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );

        // Initial access_count is 0
        let count: u32 = conn
            .query_row(
                "SELECT access_count FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        // Recall by query
        recall_by_query(
            &conn,
            &embedding_a(),
            "trackable",
            &default_filter("default"),
            &default_config(),
        )
        .unwrap();

        // access_count should be incremented
        let count: u32 = conn
            .query_row(
                "SELECT access_count FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // last_accessed should be set
        let last_accessed: Option<String> = conn
            .query_row(
                "SELECT last_accessed FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(last_accessed.is_some());
    }

    #[test]
    fn test_empty_results() {
        let conn = test_db();
        // Empty DB — should return empty, not error
        let response = recall_by_query(
            &conn,
            &embedding_a(),
            "nonexistent",
            &default_filter("default"),
            &default_config(),
        )
        .unwrap();

        assert_eq!(response.results.len(), 0);
        assert_eq!(response.total_matched, 0);
    }

    #[test]
    fn test_inspect_memory_basic() {
        let mut conn = test_db();
        let id = insert_test_memory(
            &mut conn,
            "Inspectable memory content",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            0.85,
            &embedding_a(),
        );

        let response = inspect_memory(&conn, &id, false, false).unwrap();
        assert_eq!(response.memory.id, id);
        assert_eq!(response.memory.memory_type, "semantic");
        assert_eq!(response.memory.content, "Inspectable memory content");
        assert!((response.memory.confidence - 0.85).abs() < 0.001);
        assert!(response.relations.is_none());
        assert!(response.log.is_none());
    }

    #[test]
    fn test_inspect_memory_with_log() {
        let mut conn = test_db();
        let id = insert_test_memory(
            &mut conn,
            "Memory with audit trail",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_a(),
        );

        let response = inspect_memory(&conn, &id, false, true).unwrap();
        assert!(response.log.is_some());
        let log = response.log.unwrap();
        assert!(!log.is_empty());
        assert_eq!(log[0].operation, "create");
    }

    #[test]
    fn test_inspect_memory_not_found() {
        let conn = test_db();
        let result = inspect_memory(&conn, "nonexistent-id", false, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("memory not found"));
    }

    #[test]
    fn test_truncate_preview() {
        assert_eq!(truncate_preview("short", 80), "short");
        assert_eq!(
            truncate_preview("a".repeat(100).as_str(), 80),
            format!("{}...", "a".repeat(80))
        );
    }

    #[test]
    fn test_escape_fts_query() {
        assert_eq!(escape_fts_query("hello world"), "\"hello\" \"world\"");
        assert_eq!(escape_fts_query("rust OR python"), "\"rust\" \"OR\" \"python\"");
        assert_eq!(escape_fts_query("  spaces  "), "\"spaces\"");
        assert_eq!(escape_fts_query(""), "");
    }
}
