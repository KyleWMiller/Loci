//! Write path — embedding, deduplication, storage, and audit logging.
//!
//! [`store_memory`] is the single entry point. It runs the full pipeline inside a
//! transaction: dedup check via vector similarity, insert into the memories table, sync
//! FTS5 index, insert embedding vector, handle supersession, and write an audit log.

use anyhow::{bail, Result};
use rusqlite::{params, Connection, Transaction};
use serde::Serialize;

use crate::memory::types::{MemoryType, Scope};

/// Result returned from a store operation.
#[derive(Debug, Serialize)]
pub struct StoreMemoryResult {
    /// UUID of the stored (or deduplicated) memory.
    pub id: String,
    /// Memory type as a string (e.g. `"semantic"`).
    #[serde(rename = "type")]
    pub memory_type: String,
    /// `true` if an existing near-duplicate was updated instead of creating a new record.
    pub deduplicated: bool,
    /// ID of the memory that was superseded by this one, if any.
    pub superseded: Option<String>,
}

/// Full write path: dedup check → insert or update → FTS sync → vec insert → audit log.
///
/// All operations run inside a transaction for atomicity.
pub fn store_memory(
    conn: &mut Connection,
    content: &str,
    memory_type: MemoryType,
    scope: Scope,
    group: Option<&str>,
    confidence: f64,
    metadata: Option<&serde_json::Value>,
    supersedes: Option<&str>,
    embedding: &[f32],
    dedup_threshold: f64,
) -> Result<StoreMemoryResult> {
    let tx = conn.transaction()?;

    // 1. Dedup gate
    if let Some(existing_id) = check_dedup(&tx, memory_type, embedding, dedup_threshold)? {
        update_dedup_match(&tx, &existing_id)?;
        write_audit_log(
            &tx,
            "update",
            &existing_id,
            Some(&serde_json::json!({"reason": "deduplication"})),
        )?;
        tx.commit()?;
        return Ok(StoreMemoryResult {
            id: existing_id,
            memory_type: memory_type.as_str().to_string(),
            deduplicated: true,
            superseded: None,
        });
    }

    // 2. Generate UUID v7
    let id = uuid::Uuid::now_v7().to_string();

    // 3. Insert into memories table
    let rowid = insert_memory(
        &tx,
        &id,
        memory_type,
        content,
        scope,
        group,
        confidence,
        metadata,
    )?;

    // 4. Sync FTS5 index
    insert_fts(&tx, rowid, content, &id, memory_type)?;

    // 5. Insert embedding vector
    insert_vec(&tx, &id, embedding)?;

    // 6. Handle supersession
    let superseded = if let Some(old_id) = supersedes {
        set_superseded(&tx, old_id, &id)?;
        write_audit_log(
            &tx,
            "supersede",
            old_id,
            Some(&serde_json::json!({"superseded_by": &id})),
        )?;
        Some(old_id.to_string())
    } else {
        None
    };

    // 7. Audit log for the new memory
    write_audit_log(&tx, "create", &id, None)?;

    tx.commit()?;

    Ok(StoreMemoryResult {
        id,
        memory_type: memory_type.as_str().to_string(),
        deduplicated: false,
        superseded,
    })
}

/// Check for duplicate memories of the same type with cosine similarity above threshold.
///
/// Uses sqlite-vec KNN to find nearest neighbors, then filters by type and threshold.
/// Returns `Some(existing_id)` if a duplicate is found.
fn check_dedup(
    conn: &Transaction,
    memory_type: MemoryType,
    embedding: &[f32],
    threshold: f64,
) -> Result<Option<String>> {
    let embedding_bytes = embedding_to_bytes(embedding);
    let max_distance = super::cosine_threshold_to_l2(threshold);

    let mut stmt = conn.prepare(
        "SELECT id, distance FROM memories_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT 20",
    )?;

    let candidates: Vec<(String, f64)> = stmt
        .query_map(params![embedding_bytes], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    for (candidate_id, distance) in candidates {
        // Results are ordered by distance — stop once we're past the threshold
        if distance > max_distance {
            break;
        }

        // Check if candidate has the same type and is not superseded
        let row: Option<(String, Option<String>)> = conn
            .query_row(
                "SELECT type, superseded_by FROM memories WHERE id = ?1",
                params![candidate_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        if let Some((candidate_type, superseded_by)) = row {
            if candidate_type == memory_type.as_str() && superseded_by.is_none() {
                return Ok(Some(candidate_id));
            }
        }
    }

    Ok(None)
}

/// Bump an existing memory's confidence and access count (dedup match).
fn update_dedup_match(conn: &Transaction, memory_id: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE memories SET updated_at = ?1, confidence = MIN(confidence + 0.1, 1.0), access_count = access_count + 1 WHERE id = ?2",
        params![now, memory_id],
    )?;
    Ok(())
}

/// Insert a new memory row. Returns the SQLite rowid for FTS5 sync.
fn insert_memory(
    conn: &Transaction,
    id: &str,
    memory_type: MemoryType,
    content: &str,
    scope: Scope,
    group: Option<&str>,
    confidence: f64,
    metadata: Option<&serde_json::Value>,
) -> Result<i64> {
    let now = chrono::Utc::now().to_rfc3339();
    let metadata_json = metadata.map(|m| serde_json::to_string(m)).transpose()?;

    conn.execute(
        "INSERT INTO memories (id, type, content, source_group, scope, confidence, access_count, created_at, updated_at, metadata) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7, ?7, ?8)",
        params![
            id,
            memory_type.as_str(),
            content,
            group,
            scope.as_str(),
            confidence,
            now,
            metadata_json,
        ],
    )?;

    Ok(conn.last_insert_rowid())
}

/// Sync the FTS5 index after inserting into the memories table.
///
/// Must use the same rowid as the corresponding `memories` row.
fn insert_fts(
    conn: &Transaction,
    rowid: i64,
    content: &str,
    id: &str,
    memory_type: MemoryType,
) -> Result<()> {
    conn.execute(
        "INSERT INTO memories_fts (rowid, content, id, type) VALUES (?1, ?2, ?3, ?4)",
        params![rowid, content, id, memory_type.as_str()],
    )?;
    Ok(())
}

/// Insert an embedding vector into the vec0 virtual table.
fn insert_vec(conn: &Transaction, id: &str, embedding: &[f32]) -> Result<()> {
    let embedding_bytes = embedding_to_bytes(embedding);
    conn.execute(
        "INSERT INTO memories_vec (id, embedding) VALUES (?1, ?2)",
        params![id, embedding_bytes],
    )?;
    Ok(())
}

/// Mark an old memory as superseded by a new one.
fn set_superseded(conn: &Transaction, old_id: &str, new_id: &str) -> Result<()> {
    let rows = conn.execute(
        "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
        params![new_id, old_id],
    )?;
    if rows == 0 {
        bail!("supersedes target not found: {old_id}");
    }
    Ok(())
}

/// Write an entry to the memory_log audit table.
pub(crate) fn write_audit_log(
    conn: &Connection,
    operation: &str,
    memory_id: &str,
    details: Option<&serde_json::Value>,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let details_json = details.map(|d| d.to_string());
    conn.execute(
        "INSERT INTO memory_log (operation, memory_id, details, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![operation, memory_id, details_json, now],
    )?;
    Ok(())
}

/// Re-export the shared embedding_to_bytes helper.
fn embedding_to_bytes(embedding: &[f32]) -> &[u8] {
    super::embedding_to_bytes(embedding)
}

// Import the optional extension for rusqlite
use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

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

    /// Very similar to embedding_a (cosine sim ~0.997).
    fn embedding_a_similar() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[0] = 0.99;
        v[1] = 0.07;
        // L2-normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// Orthogonal to embedding_a (cosine sim = 0.0).
    fn embedding_b() -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[100] = 1.0;
        v
    }

    #[test]
    fn test_store_new_memory() {
        let mut conn = test_db();
        let emb = embedding_a();

        let result = store_memory(
            &mut conn,
            "Rust is a systems language",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &emb,
            0.92,
        )
        .unwrap();

        assert!(!result.deduplicated);
        assert_eq!(result.memory_type, "semantic");
        assert!(result.superseded.is_none());

        // Verify in memories table
        let content: String = conn
            .query_row(
                "SELECT content FROM memories WHERE id = ?1",
                params![result.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(content, "Rust is a systems language");

        // Verify in memories_vec
        let vec_id: String = conn
            .query_row(
                "SELECT id FROM memories_vec WHERE id = ?1",
                params![result.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vec_id, result.id);

        // Verify in memories_fts
        let fts_id: String = conn
            .query_row(
                "SELECT id FROM memories_fts WHERE memories_fts MATCH 'rust'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fts_id, result.id);
    }

    #[test]
    fn test_dedup_same_type_high_similarity() {
        let mut conn = test_db();

        // Store first memory
        let result1 = store_memory(
            &mut conn,
            "Rust is great",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            0.8,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();
        assert!(!result1.deduplicated);

        // Store second with very similar embedding — should dedup
        let result2 = store_memory(
            &mut conn,
            "Rust is great indeed",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a_similar(),
            0.92,
        )
        .unwrap();

        assert!(result2.deduplicated);
        assert_eq!(result2.id, result1.id);

        // Verify confidence was boosted
        let confidence: f64 = conn
            .query_row(
                "SELECT confidence FROM memories WHERE id = ?1",
                params![result1.id],
                |row| row.get(0),
            )
            .unwrap();
        assert!((confidence - 0.9).abs() < 0.01);

        // Verify access_count was incremented
        let access_count: u32 = conn
            .query_row(
                "SELECT access_count FROM memories WHERE id = ?1",
                params![result1.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(access_count, 1);
    }

    #[test]
    fn test_dedup_different_type_no_dedup() {
        let mut conn = test_db();

        let result1 = store_memory(
            &mut conn,
            "Rust is great",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        // Same embedding but different type — should NOT dedup
        let result2 = store_memory(
            &mut conn,
            "Learning Rust today",
            MemoryType::Episodic,
            Scope::Group,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        assert!(!result2.deduplicated);
        assert_ne!(result2.id, result1.id);
    }

    #[test]
    fn test_dedup_same_type_low_similarity_no_dedup() {
        let mut conn = test_db();

        let result1 = store_memory(
            &mut conn,
            "Rust is great",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        // Orthogonal embedding — should NOT dedup
        let result2 = store_memory(
            &mut conn,
            "Python is fun",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_b(),
            0.92,
        )
        .unwrap();

        assert!(!result2.deduplicated);
        assert_ne!(result2.id, result1.id);
    }

    #[test]
    fn test_supersession() {
        let mut conn = test_db();

        let result1 = store_memory(
            &mut conn,
            "Old fact",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        let result2 = store_memory(
            &mut conn,
            "Updated fact",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            Some(&result1.id),
            &embedding_b(),
            0.92,
        )
        .unwrap();

        assert!(!result2.deduplicated);
        assert_eq!(result2.superseded.as_deref(), Some(result1.id.as_str()));

        // Verify old memory has superseded_by set
        let superseded_by: Option<String> = conn
            .query_row(
                "SELECT superseded_by FROM memories WHERE id = ?1",
                params![result1.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(superseded_by.as_deref(), Some(result2.id.as_str()));
    }

    #[test]
    fn test_audit_log_written() {
        let mut conn = test_db();

        let result = store_memory(
            &mut conn,
            "Test memory",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        let (op, mid): (String, String) = conn
            .query_row(
                "SELECT operation, memory_id FROM memory_log WHERE memory_id = ?1",
                params![result.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(op, "create");
        assert_eq!(mid, result.id);
    }

    #[test]
    fn test_confidence_cap() {
        let mut conn = test_db();

        // Store with confidence 0.95
        let result1 = store_memory(
            &mut conn,
            "Capped confidence",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            0.95,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        // Dedup — should boost to 1.0 (capped), not 1.05
        let _ = store_memory(
            &mut conn,
            "Capped confidence again",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a_similar(),
            0.92,
        )
        .unwrap();

        let confidence: f64 = conn
            .query_row(
                "SELECT confidence FROM memories WHERE id = ?1",
                params![result1.id],
                |row| row.get(0),
            )
            .unwrap();
        assert!((confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fts_search() {
        let mut conn = test_db();

        store_memory(
            &mut conn,
            "The quantum computer operates at very low temperatures",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        let found: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM memories_fts WHERE memories_fts MATCH 'quantum'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(found);
    }

    #[test]
    fn test_supersedes_nonexistent_fails() {
        let mut conn = test_db();

        let result = store_memory(
            &mut conn,
            "Replacing nothing",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            Some("nonexistent-id"),
            &embedding_a(),
            0.92,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("supersedes target not found")
        );
    }

    #[test]
    fn test_dedup_skips_superseded_memories() {
        let mut conn = test_db();

        // Store memory A
        let result1 = store_memory(
            &mut conn,
            "Original fact",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap();

        // Supersede A with B (different embedding so no dedup)
        let result2 = store_memory(
            &mut conn,
            "Updated fact",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            Some(&result1.id),
            &embedding_b(),
            0.92,
        )
        .unwrap();
        assert_eq!(result2.superseded.as_deref(), Some(result1.id.as_str()));

        // Store C with same embedding as A — should NOT dedup against A (it's superseded)
        let result3 = store_memory(
            &mut conn,
            "Another similar fact",
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a_similar(),
            0.92,
        )
        .unwrap();

        assert!(!result3.deduplicated);
        assert_ne!(result3.id, result1.id);
    }
}
