//! Soft and hard memory deletion.
//!
//! Soft delete marks a memory as superseded (by "forgotten"); hard delete permanently
//! removes it from the memories table, FTS5 index, vector index, and cascades to relations.

use anyhow::{bail, Result};
use rusqlite::{params, Connection};
use serde::Serialize;

use super::store::write_audit_log;

/// Result returned from a forget operation.
#[derive(Debug, Serialize)]
pub struct ForgetResult {
    /// ID of the forgotten memory.
    pub id: String,
    /// `true` if the memory was permanently removed; `false` for soft delete.
    pub hard_deleted: bool,
}

/// Forget a memory by ID.
///
/// Soft delete (default): sets `superseded_by = "forgotten"` and logs reason.
/// Hard delete: removes from all tables (memories, FTS, vec), cascades entity_relations via FK.
pub fn forget_memory(
    conn: &mut Connection,
    memory_id: &str,
    reason: Option<&str>,
    hard_delete: bool,
) -> Result<ForgetResult> {
    if hard_delete {
        hard_delete_memory(conn, memory_id, reason)
    } else {
        soft_delete_memory(conn, memory_id, reason)
    }
}

/// Soft delete: mark as superseded by "forgotten".
fn soft_delete_memory(
    conn: &mut Connection,
    memory_id: &str,
    reason: Option<&str>,
) -> Result<ForgetResult> {
    let tx = conn.transaction()?;

    // Verify memory exists
    let exists: bool = tx.query_row(
        "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
        params![memory_id],
        |row| row.get(0),
    )?;
    if !exists {
        bail!("memory not found: {memory_id}");
    }

    // Set superseded_by to "forgotten"
    tx.execute(
        "UPDATE memories SET superseded_by = 'forgotten', updated_at = ?1 WHERE id = ?2",
        params![chrono::Utc::now().to_rfc3339(), memory_id],
    )?;

    // Audit log
    let details = serde_json::json!({
        "reason": reason,
        "hard_delete": false,
    });
    write_audit_log(&tx, "delete", memory_id, Some(&details))?;

    tx.commit()?;

    Ok(ForgetResult {
        id: memory_id.to_string(),
        hard_deleted: false,
    })
}

/// Hard delete: remove from all tables.
fn hard_delete_memory(
    conn: &mut Connection,
    memory_id: &str,
    reason: Option<&str>,
) -> Result<ForgetResult> {
    let tx = conn.transaction()?;

    // Fetch rowid, content, and type for FTS5 cleanup
    let (rowid, content, memory_type): (i64, String, String) = tx
        .query_row(
            "SELECT rowid, content, type FROM memories WHERE id = ?1",
            params![memory_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                anyhow::anyhow!("memory not found: {memory_id}")
            }
            other => anyhow::anyhow!("database error: {other}"),
        })?;

    // 1. Remove from FTS5 index (external content table requires special delete)
    tx.execute(
        "INSERT INTO memories_fts(memories_fts, rowid, content, id, type) VALUES('delete', ?1, ?2, ?3, ?4)",
        params![rowid, content, memory_id, memory_type],
    )?;

    // 2. Remove from vector table
    tx.execute(
        "DELETE FROM memories_vec WHERE id = ?1",
        params![memory_id],
    )?;

    // 3. Audit log (before deleting memory row, since we reference memory_id as text)
    let details = serde_json::json!({
        "reason": reason,
        "hard_delete": true,
    });
    write_audit_log(&tx, "delete", memory_id, Some(&details))?;

    // 4. Delete from memories (cascades to entity_relations via FK)
    tx.execute("DELETE FROM memories WHERE id = ?1", params![memory_id])?;

    tx.commit()?;

    Ok(ForgetResult {
        id: memory_id.to_string(),
        hard_deleted: true,
    })
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

    fn insert_memory(conn: &mut Connection, content: &str, emb: &[f32]) -> String {
        store::store_memory(
            conn,
            content,
            MemoryType::Semantic,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            emb,
            0.92,
        )
        .unwrap()
        .id
    }

    #[test]
    fn test_soft_delete_sets_superseded() {
        let mut conn = test_db();
        let id = insert_memory(&mut conn, "To be soft deleted", &embedding_a());

        let result = forget_memory(&mut conn, &id, Some("outdated"), false).unwrap();
        assert_eq!(result.id, id);
        assert!(!result.hard_deleted);

        // Verify superseded_by is set
        let superseded: String = conn
            .query_row(
                "SELECT superseded_by FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(superseded, "forgotten");
    }

    #[test]
    fn test_soft_delete_writes_audit_log() {
        let mut conn = test_db();
        let id = insert_memory(&mut conn, "Audit test", &embedding_a());

        forget_memory(&mut conn, &id, Some("test reason"), false).unwrap();

        let (op, details_str): (String, String) = conn
            .query_row(
                "SELECT operation, details FROM memory_log WHERE memory_id = ?1 AND operation = 'delete'",
                params![id],
                |row| Ok((row.get(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        assert_eq!(op, "delete");
        let details: serde_json::Value = serde_json::from_str(&details_str).unwrap();
        assert_eq!(details["reason"], "test reason");
        assert_eq!(details["hard_delete"], false);
    }

    #[test]
    fn test_hard_delete_removes_from_all_tables() {
        let mut conn = test_db();
        let id = insert_memory(&mut conn, "To be hard deleted", &embedding_a());

        let result = forget_memory(&mut conn, &id, None, true).unwrap();
        assert_eq!(result.id, id);
        assert!(result.hard_deleted);

        // Verify removed from memories
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        // Verify removed from memories_vec
        let vec_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_vec WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vec_count, 0);

        // Verify FTS is clean (search should not find it)
        let fts_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"hard deleted\"'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fts_count, 0);
    }

    #[test]
    fn test_hard_delete_cascades_relations() {
        let mut conn = test_db();

        // Create two entity memories
        let id_a = store::store_memory(
            &mut conn,
            "Entity A",
            MemoryType::Entity,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_a(),
            0.92,
        )
        .unwrap()
        .id;
        let id_b = store::store_memory(
            &mut conn,
            "Entity B",
            MemoryType::Entity,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            &embedding_b(),
            0.92,
        )
        .unwrap()
        .id;

        // Create a relation
        crate::memory::relations::store_relation(&conn, &id_a, "knows", &id_b).unwrap();

        // Hard delete entity A
        forget_memory(&mut conn, &id_a, None, true).unwrap();

        // Relation should be gone (FK cascade)
        let rel_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entity_relations WHERE subject_id = ?1 OR object_id = ?1",
                params![id_a],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rel_count, 0);
    }

    #[test]
    fn test_hard_delete_writes_audit_log() {
        let mut conn = test_db();
        let id = insert_memory(&mut conn, "Hard delete audit", &embedding_a());

        forget_memory(&mut conn, &id, Some("no longer needed"), true).unwrap();

        // Audit log entry should still exist (memory_log has no FK to memories)
        let (op, details_str): (String, String) = conn
            .query_row(
                "SELECT operation, details FROM memory_log WHERE memory_id = ?1 AND operation = 'delete'",
                params![id],
                |row| Ok((row.get(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        assert_eq!(op, "delete");
        let details: serde_json::Value = serde_json::from_str(&details_str).unwrap();
        assert_eq!(details["hard_delete"], true);
        assert_eq!(details["reason"], "no longer needed");
    }

    #[test]
    fn test_forget_nonexistent_memory_fails() {
        let mut conn = test_db();

        let result = forget_memory(&mut conn, "nonexistent-id", None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("memory not found"));

        let result = forget_memory(&mut conn, "nonexistent-id", None, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("memory not found"));
    }
}
