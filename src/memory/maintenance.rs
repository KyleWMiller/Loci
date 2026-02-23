use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use super::store::write_audit_log;
use crate::config::MaintenanceConfig;
use crate::embedding::EmbeddingProvider;

// ── Result types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DecayResult {
    pub affected_by_type: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct CompactResult {
    pub groups_compacted: usize,
    pub memories_compacted: usize,
    pub summaries_created: usize,
}

#[derive(Debug, Serialize)]
pub struct PromoteResult {
    pub clusters_found: usize,
    pub semantics_created: usize,
}

#[derive(Debug, Serialize)]
pub struct CleanupResult {
    pub candidates: Vec<CleanupCandidate>,
    pub deleted: usize,
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
pub struct CleanupCandidate {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub confidence: f64,
    pub content_preview: String,
    pub last_accessed: Option<String>,
    pub created_at: String,
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Row for an episodic memory eligible for compaction.
struct EpisodicRow {
    id: String,
    content: String,
    source_group: Option<String>,
    scope: String,
    /// ISO year-week string like "2026-W08"
    week_key: String,
}

/// Re-export the shared cosine-to-L2 conversion.
fn cosine_threshold_to_l2(cosine_threshold: f64) -> f64 {
    super::cosine_threshold_to_l2(cosine_threshold)
}

/// Truncate content to max_chars, appending "..." if truncated.
fn truncate(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        let end = content
            .char_indices()
            .take_while(|(i, _)| *i < max_chars)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(max_chars);
        format!("{}...", &content[..end])
    }
}

// ── Confidence Decay ─────────────────────────────────────────────────────────

/// Apply confidence decay to all active memories, per-type.
///
/// Episodic memories decay faster (default 0.95) than semantic/procedural/entity (0.99).
/// Only non-superseded memories with confidence > 0 are affected.
pub fn apply_decay(conn: &Connection, config: &MaintenanceConfig) -> Result<DecayResult> {
    let now = chrono::Utc::now().to_rfc3339();
    let mut affected_by_type = HashMap::new();

    let type_factors = [
        ("episodic", config.episodic_decay_factor),
        ("semantic", config.semantic_decay_factor),
        ("procedural", config.semantic_decay_factor),
        ("entity", config.semantic_decay_factor),
    ];

    for (memory_type, factor) in &type_factors {
        let affected = conn.execute(
            "UPDATE memories SET confidence = confidence * ?1, updated_at = ?2 \
             WHERE type = ?3 AND superseded_by IS NULL AND confidence > 0.0",
            params![factor, now, memory_type],
        )?;

        if affected > 0 {
            // Use a synthetic memory_id for decay audit entries (batch operation)
            write_audit_log(
                conn,
                "decay",
                &format!("batch:{memory_type}"),
                Some(&serde_json::json!({
                    "type": memory_type,
                    "factor": factor,
                    "affected": affected,
                })),
            )?;
        }

        affected_by_type.insert(memory_type.to_string(), affected);
    }

    Ok(DecayResult { affected_by_type })
}

// ── Episodic Compaction ──────────────────────────────────────────────────────

/// Compact old episodic memories by grouping them by week + source_group,
/// concatenating their content, and creating a summary memory.
///
/// Originals are superseded by the new summary.
pub fn compact_episodic(
    conn: &mut Connection,
    embedding_provider: &dyn EmbeddingProvider,
    config: &MaintenanceConfig,
) -> Result<CompactResult> {
    let cutoff = chrono::Utc::now() - chrono::Duration::days(config.compaction_age_days as i64);
    let cutoff_str = cutoff.to_rfc3339();

    // Fetch qualifying episodic memories (scoped to drop stmt before mutable ops)
    let rows: Vec<EpisodicRow> = {
        let mut stmt = conn.prepare(
            "SELECT id, content, source_group, scope, \
             strftime('%Y-W%W', created_at) as week_key \
             FROM memories \
             WHERE type = 'episodic' \
               AND superseded_by IS NULL \
               AND created_at < ?1 \
             ORDER BY source_group, week_key, created_at",
        )?;
        let collected = stmt
            .query_map(params![cutoff_str], |row| {
                Ok(EpisodicRow {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    source_group: row.get(2)?,
                    scope: row.get(3)?,
                    week_key: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        collected
    };

    // Group by (source_group, week_key)
    let mut groups: HashMap<(Option<String>, String), Vec<EpisodicRow>> = HashMap::new();
    for row in rows {
        let key = (row.source_group.clone(), row.week_key.clone());
        groups.entry(key).or_default().push(row);
    }

    let mut result = CompactResult {
        groups_compacted: 0,
        memories_compacted: 0,
        summaries_created: 0,
    };

    for ((_group, _week), members) in &groups {
        if members.len() < config.compaction_min_group_size {
            continue;
        }

        // Concatenate content
        let combined: String = members
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        let summary_content = truncate(&combined, 4000);

        // Embed the summary
        let embedding = embedding_provider.embed(&summary_content)?;

        // Determine group/scope from first member
        let group = members[0].source_group.as_deref();
        let scope = match members[0].scope.as_str() {
            "group" => crate::memory::types::Scope::Group,
            _ => crate::memory::types::Scope::Global,
        };

        let metadata = serde_json::json!({"summary": true});

        // Store the summary memory (dedup threshold set high to avoid matching)
        let store_result = super::store::store_memory(
            conn,
            &summary_content,
            crate::memory::types::MemoryType::Episodic,
            scope,
            group,
            1.0,
            Some(&metadata),
            None,
            &embedding,
            0.99, // high threshold to avoid dedup against existing
        )?;

        // Supersede all originals
        let tx = conn.transaction()?;
        for member in members {
            tx.execute(
                "UPDATE memories SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params![store_result.id, chrono::Utc::now().to_rfc3339(), member.id],
            )?;
        }
        write_audit_log(
            &tx,
            "compact",
            &store_result.id,
            Some(&serde_json::json!({
                "source_count": members.len(),
                "summary_id": store_result.id,
            })),
        )?;
        tx.commit()?;

        result.groups_compacted += 1;
        result.memories_compacted += members.len();
        result.summaries_created += 1;
    }

    Ok(result)
}

// ── Episodic-to-Semantic Promotion ───────────────────────────────────────────

/// Find clusters of similar episodic memories and promote them to semantic.
///
/// Episodic memories with cosine similarity > promotion_similarity that appear
/// in clusters of >= promotion_threshold are distilled into a semantic memory.
/// The episodic sources are NOT superseded (they retain event context).
pub fn promote_episodic_to_semantic(
    conn: &mut Connection,
    embedding_provider: &dyn EmbeddingProvider,
    config: &MaintenanceConfig,
) -> Result<PromoteResult> {
    struct EpisodicCandidate {
        id: String,
        content: String,
        access_count: u32,
        embedding: Vec<f32>,
    }

    // Fetch all non-superseded episodic memories (scoped to drop stmt)
    let candidates: Vec<EpisodicCandidate> = {
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.access_count, v.embedding \
             FROM memories m \
             JOIN memories_vec v ON m.id = v.id \
             WHERE m.type = 'episodic' AND m.superseded_by IS NULL",
        )?;
        let collected = stmt
            .query_map([], |row| {
                let embedding_bytes: Vec<u8> = row.get(3)?;
                let embedding = bytes_to_embedding(&embedding_bytes);
                Ok(EpisodicCandidate {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    access_count: row.get(2)?,
                    embedding,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        collected
    };

    let mut processed: HashSet<String> = HashSet::new();
    let mut result = PromoteResult {
        clusters_found: 0,
        semantics_created: 0,
    };

    let max_distance = cosine_threshold_to_l2(config.promotion_similarity);

    for candidate in &candidates {
        if processed.contains(&candidate.id) {
            continue;
        }

        // Find similar episodic memories and build cluster (scoped to drop stmts)
        let cluster_ids: Vec<String> = {
            let embedding_bytes = super::embedding_to_bytes(&candidate.embedding);
            let mut knn_stmt = conn.prepare(
                "SELECT id, distance FROM memories_vec \
                 WHERE embedding MATCH ?1 ORDER BY distance LIMIT 50",
            )?;
            let neighbors: Vec<(String, f64)> = knn_stmt
                .query_map(params![embedding_bytes], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;

            // Collect neighbor IDs within similarity threshold
            let mut neighbor_ids: Vec<String> = Vec::new();
            for (neighbor_id, distance) in &neighbors {
                if *distance > max_distance {
                    break;
                }
                if !processed.contains(neighbor_id) {
                    neighbor_ids.push(neighbor_id.clone());
                }
            }
            neighbor_ids
        };

        // Filter to episodic, non-superseded
        let mut eligible_ids: Vec<String> = Vec::new();
        for neighbor_id in &cluster_ids {
            let is_eligible: bool = conn
                .query_row(
                    "SELECT type = 'episodic' AND superseded_by IS NULL \
                     FROM memories WHERE id = ?1",
                    params![neighbor_id],
                    |row| row.get(0),
                )
                .unwrap_or(false);
            if is_eligible {
                eligible_ids.push(neighbor_id.clone());
            }
        }

        if eligible_ids.len() < config.promotion_threshold {
            continue;
        }

        result.clusters_found += 1;

        // Pick the most-accessed memory's content as the distilled fact
        let best = candidates
            .iter()
            .filter(|c| eligible_ids.contains(&c.id))
            .max_by_key(|c| c.access_count)
            .unwrap_or(candidate);

        // Embed the distilled fact
        let embedding = embedding_provider.embed(&best.content)?;

        // Store as semantic memory (dedup gate will catch existing similar semantics)
        let store_result = super::store::store_memory(
            conn,
            &best.content,
            crate::memory::types::MemoryType::Semantic,
            crate::memory::types::Scope::Global,
            None,
            1.0,
            Some(&serde_json::json!({"promoted_from": "episodic"})),
            None,
            &embedding,
            config.promotion_similarity,
        )?;

        if !store_result.deduplicated {
            write_audit_log(
                conn,
                "compact",
                &store_result.id,
                Some(&serde_json::json!({
                    "action": "promote",
                    "source_count": eligible_ids.len(),
                    "semantic_id": store_result.id,
                })),
            )?;
            result.semantics_created += 1;
        }

        // Mark all cluster members as processed (don't re-promote)
        for id in &eligible_ids {
            processed.insert(id.clone());
        }
    }

    Ok(result)
}

/// Convert raw bytes back to f32 embedding.
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

/// Find and optionally delete stale, low-confidence memories.
///
/// Candidates: confidence < floor AND (never accessed and old, OR last accessed long ago).
/// In dry_run mode, returns candidates without deleting.
pub fn cleanup_stale(
    conn: &mut Connection,
    config: &MaintenanceConfig,
    dry_run: bool,
) -> Result<CleanupResult> {
    let threshold =
        chrono::Utc::now() - chrono::Duration::days(config.cleanup_no_access_days as i64);
    let threshold_str = threshold.to_rfc3339();

    let candidates: Vec<CleanupCandidate> = {
        let mut stmt = conn.prepare(
            "SELECT id, type, confidence, content, last_accessed, created_at \
             FROM memories \
             WHERE superseded_by IS NULL \
               AND confidence < ?1 \
               AND ( \
                   (last_accessed IS NULL AND created_at < ?2) \
                   OR (last_accessed IS NOT NULL AND last_accessed < ?2) \
               )",
        )?;
        let collected = stmt
            .query_map(params![config.cleanup_confidence_floor, threshold_str], |row| {
                let content: String = row.get(3)?;
                Ok(CleanupCandidate {
                    id: row.get(0)?,
                    memory_type: row.get(1)?,
                    confidence: row.get(2)?,
                    content_preview: truncate(&content, 80),
                    last_accessed: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        collected
    };

    if dry_run {
        return Ok(CleanupResult {
            deleted: 0,
            dry_run: true,
            candidates,
        });
    }

    let mut deleted = 0;
    for candidate in &candidates {
        hard_delete_memory(conn, &candidate.id)?;
        deleted += 1;
    }

    Ok(CleanupResult {
        deleted,
        dry_run: false,
        candidates,
    })
}

/// Hard delete a single memory from all tables (memories, FTS, vec).
///
/// Replicates the pattern from forget.rs but without the existence check
/// (caller already verified the row exists via the candidate query).
fn hard_delete_memory(conn: &mut Connection, memory_id: &str) -> Result<()> {
    let tx = conn.transaction()?;

    // Fetch rowid, content, type for FTS cleanup
    let (rowid, content, memory_type): (i64, String, String) = tx.query_row(
        "SELECT rowid, content, type FROM memories WHERE id = ?1",
        params![memory_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;

    // Remove from FTS5 (external content table requires special delete syntax)
    tx.execute(
        "INSERT INTO memories_fts(memories_fts, rowid, content, id, type) VALUES('delete', ?1, ?2, ?3, ?4)",
        params![rowid, content, memory_id, memory_type],
    )?;

    // Remove from vector table
    tx.execute(
        "DELETE FROM memories_vec WHERE id = ?1",
        params![memory_id],
    )?;

    // Audit log
    write_audit_log(
        &tx,
        "delete",
        memory_id,
        Some(&serde_json::json!({"reason": "cleanup", "hard_delete": true})),
    )?;

    // Delete from memories (cascades entity_relations via FK)
    tx.execute("DELETE FROM memories WHERE id = ?1", params![memory_id])?;

    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::store;
    use crate::memory::types::{MemoryType, Scope};

    fn test_db() -> Connection {
        db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::db::schema::init_schema(&conn).unwrap();
        conn
    }

    fn default_config() -> MaintenanceConfig {
        MaintenanceConfig::default()
    }

    fn embedding_a() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[0] = 1.0;
        v
    }

    fn embedding_b() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[100] = 1.0;
        v
    }

    fn insert_memory(
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
            0.99, // high threshold to avoid test dedup
        )
        .unwrap()
        .id
    }

    /// Insert a memory with a backdated created_at timestamp.
    fn insert_old_memory(
        conn: &mut Connection,
        content: &str,
        memory_type: MemoryType,
        group: &str,
        confidence: f64,
        embedding: &[f32],
        days_ago: i64,
    ) -> String {
        let id = insert_memory(conn, content, memory_type, Scope::Group, group, confidence, embedding);
        let old_date = (chrono::Utc::now() - chrono::Duration::days(days_ago)).to_rfc3339();
        conn.execute(
            "UPDATE memories SET created_at = ?1, updated_at = ?1 WHERE id = ?2",
            params![old_date, id],
        )
        .unwrap();
        id
    }

    // ── Decay tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_decay_applies_per_type() {
        let mut conn = test_db();
        let config = default_config();

        let id_epi = insert_memory(
            &mut conn,
            "Episodic event",
            MemoryType::Episodic,
            Scope::Group,
            "default",
            1.0,
            &embedding_a(),
        );
        let id_sem = insert_memory(
            &mut conn,
            "Semantic fact",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            1.0,
            &embedding_b(),
        );

        apply_decay(&conn, &config).unwrap();

        let epi_conf: f64 = conn
            .query_row(
                "SELECT confidence FROM memories WHERE id = ?1",
                params![id_epi],
                |row| row.get(0),
            )
            .unwrap();
        let sem_conf: f64 = conn
            .query_row(
                "SELECT confidence FROM memories WHERE id = ?1",
                params![id_sem],
                |row| row.get(0),
            )
            .unwrap();

        // Episodic decays by 0.95, semantic by 0.99
        assert!((epi_conf - 0.95).abs() < 0.001);
        assert!((sem_conf - 0.99).abs() < 0.001);
        // Episodic decayed more
        assert!(epi_conf < sem_conf);
    }

    #[test]
    fn test_decay_skips_superseded() {
        let mut conn = test_db();
        let config = default_config();

        let id = insert_memory(
            &mut conn,
            "Superseded memory",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            0.8,
            &embedding_a(),
        );

        // Mark as superseded
        conn.execute(
            "UPDATE memories SET superseded_by = 'some-id' WHERE id = ?1",
            params![id],
        )
        .unwrap();

        apply_decay(&conn, &config).unwrap();

        let conf: f64 = conn
            .query_row(
                "SELECT confidence FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        // Should not have changed
        assert!((conf - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_decay_audit_log() {
        let mut conn = test_db();
        let config = default_config();

        insert_memory(
            &mut conn,
            "Memory for audit",
            MemoryType::Episodic,
            Scope::Group,
            "default",
            1.0,
            &embedding_a(),
        );

        apply_decay(&conn, &config).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_log WHERE operation = 'decay'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count > 0);
    }

    // ── Cleanup tests ────────────────────────────────────────────────────────

    #[test]
    fn test_cleanup_stale_dry_run() {
        let mut conn = test_db();
        let config = default_config();

        // Insert a low-confidence, old memory
        let id = insert_old_memory(
            &mut conn,
            "Stale memory",
            MemoryType::Semantic,
            "default",
            0.01,
            &embedding_a(),
            120, // 120 days ago
        );

        let result = cleanup_stale(&mut conn, &config, true).unwrap();
        assert!(result.dry_run);
        assert_eq!(result.deleted, 0);
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].id, id);

        // Verify memory still exists
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_cleanup_stale_hard_delete() {
        let mut conn = test_db();
        let config = default_config();

        let id = insert_old_memory(
            &mut conn,
            "Stale to delete",
            MemoryType::Semantic,
            "default",
            0.01,
            &embedding_a(),
            120,
        );

        let result = cleanup_stale(&mut conn, &config, false).unwrap();
        assert!(!result.dry_run);
        assert_eq!(result.deleted, 1);

        // Verify memory is gone from all tables
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        let vec_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_vec WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vec_count, 0);
    }

    #[test]
    fn test_cleanup_skips_recent() {
        let mut conn = test_db();
        let config = default_config();

        // Low confidence but recent
        insert_memory(
            &mut conn,
            "Recent low confidence",
            MemoryType::Semantic,
            Scope::Global,
            "default",
            0.01,
            &embedding_a(),
        );

        let result = cleanup_stale(&mut conn, &config, true).unwrap();
        assert_eq!(result.candidates.len(), 0);
    }

    #[test]
    fn test_cleanup_skips_high_confidence() {
        let mut conn = test_db();
        let config = default_config();

        // Old but high confidence
        insert_old_memory(
            &mut conn,
            "Old but confident",
            MemoryType::Semantic,
            "default",
            0.5,
            &embedding_a(),
            120,
        );

        let result = cleanup_stale(&mut conn, &config, true).unwrap();
        assert_eq!(result.candidates.len(), 0);
    }

    // ── Compaction tests ─────────────────────────────────────────────────────

    /// Test embedding provider that returns a fixed embedding.
    struct TestEmbeddingProvider;

    impl EmbeddingProvider for TestEmbeddingProvider {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            // Return a unique embedding based on text hash to avoid dedup
            let mut v = vec![0.0f32; 384];
            let hash = _text.len() % 384;
            v[hash] = 1.0;
            Ok(v)
        }
    }

    #[test]
    fn test_compact_groups_by_week() {
        let mut conn = test_db();
        let mut config = default_config();
        config.compaction_min_group_size = 3;

        // Insert 4 old episodic memories (same group, will share a week)
        for i in 0..4 {
            let mut emb = vec![0.0f32; 384];
            emb[i + 1] = 1.0; // unique embeddings
            insert_old_memory(
                &mut conn,
                &format!("Episodic event {i} from the past"),
                MemoryType::Episodic,
                "project-a",
                1.0,
                &emb,
                45, // same day, 45 days ago
            );
        }

        let result =
            compact_episodic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        assert_eq!(result.groups_compacted, 1);
        assert_eq!(result.memories_compacted, 4);
        assert_eq!(result.summaries_created, 1);

        // Originals should be superseded
        let superseded_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE type = 'episodic' AND superseded_by IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(superseded_count, 4);
    }

    #[test]
    fn test_compact_skips_small_groups() {
        let mut conn = test_db();
        let mut config = default_config();
        config.compaction_min_group_size = 5;

        // Insert only 3 old episodic memories (below threshold)
        for i in 0..3 {
            let mut emb = vec![0.0f32; 384];
            emb[i + 1] = 1.0;
            insert_old_memory(
                &mut conn,
                &format!("Small group event {i}"),
                MemoryType::Episodic,
                "project-b",
                1.0,
                &emb,
                45,
            );
        }

        let result =
            compact_episodic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        assert_eq!(result.groups_compacted, 0);
        assert_eq!(result.memories_compacted, 0);
    }

    #[test]
    fn test_compact_supersedes_originals() {
        let mut conn = test_db();
        let mut config = default_config();
        config.compaction_min_group_size = 2;

        let ids: Vec<String> = (0..3)
            .map(|i| {
                let mut emb = vec![0.0f32; 384];
                emb[i + 1] = 1.0;
                insert_old_memory(
                    &mut conn,
                    &format!("Compactable event {i}"),
                    MemoryType::Episodic,
                    "project-c",
                    1.0,
                    &emb,
                    45,
                )
            })
            .collect();

        compact_episodic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        // All originals should have superseded_by set to the same summary ID
        let superseded_bys: Vec<String> = ids
            .iter()
            .map(|id| {
                conn.query_row(
                    "SELECT superseded_by FROM memories WHERE id = ?1",
                    params![id],
                    |row| row.get::<_, String>(0),
                )
                .unwrap()
            })
            .collect();

        // All should point to the same summary
        assert!(superseded_bys.iter().all(|s| s == &superseded_bys[0]));

        // Summary should exist and have metadata.summary = true
        let metadata_str: String = conn
            .query_row(
                "SELECT metadata FROM memories WHERE id = ?1",
                params![superseded_bys[0]],
                |row| row.get(0),
            )
            .unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
        assert_eq!(metadata["summary"], true);
    }

    // ── Promotion tests ──────────────────────────────────────────────────────

    #[test]
    fn test_promotion_creates_semantic() {
        let mut conn = test_db();
        let mut config = default_config();
        config.promotion_threshold = 3;
        config.promotion_similarity = 0.88;

        // Insert 3 episodic memories with embeddings that are:
        // - similar enough for promotion (pairwise cosine sim > 0.88)
        // - different enough to avoid dedup (pairwise cosine sim < 0.99)
        // Spread perturbations across different secondary dimensions to avoid dedup
        let embeddings: Vec<Vec<f32>> = vec![
            {
                let mut v = vec![0.0f32; 384];
                v[0] = 1.0;
                v // [1, 0, 0, ...] — cosine sim ~0.95 with others
            },
            {
                let mut v = vec![0.0f32; 384];
                v[0] = 0.95;
                v[1] = 0.31;
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            },
            {
                let mut v = vec![0.0f32; 384];
                v[0] = 0.95;
                v[2] = 0.31; // different secondary dimension
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            },
        ];
        for (i, emb) in embeddings.iter().enumerate() {
            insert_memory(
                &mut conn,
                &format!("Similar episodic fact #{i}"),
                MemoryType::Episodic,
                Scope::Group,
                "default",
                1.0,
                emb,
            );
        }

        let result =
            promote_episodic_to_semantic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        assert_eq!(result.clusters_found, 1);
        assert_eq!(result.semantics_created, 1);

        // Verify a semantic memory was created
        let sem_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE type = 'semantic'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sem_count, 1);

        // Episodics should NOT be superseded
        let epi_superseded: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE type = 'episodic' AND superseded_by IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(epi_superseded, 0);
    }

    #[test]
    fn test_promotion_skips_below_threshold() {
        let mut conn = test_db();
        let mut config = default_config();
        config.promotion_threshold = 5;

        // Insert only 2 similar episodics (below threshold of 5)
        for i in 0..2 {
            let mut emb = embedding_a();
            emb[1] = 0.01 * i as f32;
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            emb.iter_mut().for_each(|x| *x /= norm);

            insert_memory(
                &mut conn,
                &format!("Too few similar #{i}"),
                MemoryType::Episodic,
                Scope::Group,
                "default",
                1.0,
                &emb,
            );
        }

        let result =
            promote_episodic_to_semantic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        assert_eq!(result.clusters_found, 0);
        assert_eq!(result.semantics_created, 0);
    }

    #[test]
    fn test_promotion_does_not_repromote() {
        let mut conn = test_db();
        let mut config = default_config();
        config.promotion_threshold = 3;
        config.promotion_similarity = 0.88;

        // Insert 4 episodic memories: similar enough for promotion, different enough to avoid dedup
        // Each uses a different secondary dimension to avoid pairwise dedup
        let embeddings: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                let mut v = vec![0.0f32; 384];
                v[0] = 0.95;
                v[i + 1] = 0.31; // unique secondary dimension per memory
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            })
            .collect();
        for (i, emb) in embeddings.iter().enumerate() {
            insert_memory(
                &mut conn,
                &format!("Repeated fact variant #{i}"),
                MemoryType::Episodic,
                Scope::Group,
                "default",
                1.0,
                emb,
            );
        }

        // Run promotion — should create exactly 1 semantic (not multiple for overlapping clusters)
        let result =
            promote_episodic_to_semantic(&mut conn, &TestEmbeddingProvider, &config).unwrap();

        assert_eq!(result.clusters_found, 1);
        assert_eq!(result.semantics_created, 1);
    }
}
