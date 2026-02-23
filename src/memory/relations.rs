//! Entity relationship storage and deduplication.
//!
//! Stores directed (subject, predicate, object) triples between entity-type memories,
//! with automatic deduplication on the full triple.

use anyhow::{bail, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

/// Result returned from a store_relation operation.
#[derive(Debug, Serialize)]
pub struct StoreRelationResult {
    /// UUID of the created (or existing) relation.
    pub id: String,
    /// `true` if this exact (subject, predicate, object) triple already existed.
    pub deduplicated: bool,
}

/// Store a relationship between two entity memories.
///
/// Validates both IDs exist and are entity-type. Deduplicates on the
/// (subject_id, predicate, object_id) tuple — storing the same relation
/// twice is idempotent.
pub fn store_relation(
    conn: &Connection,
    subject_id: &str,
    predicate: &str,
    object_id: &str,
) -> Result<StoreRelationResult> {
    // Validate subject exists and is entity type
    validate_entity(conn, subject_id, "subject")?;

    // Validate object exists and is entity type
    validate_entity(conn, object_id, "object")?;

    // Dedup: check for existing (subject, predicate, object) tuple
    let existing_id: Option<String> = conn
        .query_row(
            "SELECT id FROM entity_relations \
             WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3",
            params![subject_id, predicate, object_id],
            |row| row.get(0),
        )
        .optional()?;

    if let Some(id) = existing_id {
        return Ok(StoreRelationResult {
            id,
            deduplicated: true,
        });
    }

    // Insert new relation
    let id = uuid::Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO entity_relations (id, subject_id, predicate, object_id, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![id, subject_id, predicate, object_id, now],
    )?;

    Ok(StoreRelationResult {
        id,
        deduplicated: false,
    })
}

/// Validate that a memory ID exists and is entity type.
fn validate_entity(conn: &Connection, memory_id: &str, role: &str) -> Result<()> {
    let row: Option<String> = conn
        .query_row(
            "SELECT type FROM memories WHERE id = ?1",
            params![memory_id],
            |row| row.get(0),
        )
        .optional()?;

    match row {
        None => bail!("{role} memory not found: {memory_id}"),
        Some(t) if t != "entity" => {
            bail!("{role} memory must be entity type, got: {t}")
        }
        Some(_) => Ok(()),
    }
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

    /// Helper: insert an entity memory and return its ID.
    fn insert_entity(conn: &mut Connection, content: &str, embedding: &[f32]) -> String {
        store::store_memory(
            conn,
            content,
            MemoryType::Entity,
            Scope::Global,
            Some("default"),
            1.0,
            None,
            None,
            embedding,
            0.92,
        )
        .unwrap()
        .id
    }

    #[test]
    fn test_store_relation_basic() {
        let mut conn = test_db();
        let id_a = insert_entity(&mut conn, "John Smith is an engineer", &embedding_a());
        let id_b = insert_entity(&mut conn, "Acme Corp is a company", &embedding_b());

        let result = store_relation(&conn, &id_a, "works_at", &id_b).unwrap();
        assert!(!result.deduplicated);

        // Verify in DB
        let (subj, pred, obj): (String, String, String) = conn
            .query_row(
                "SELECT subject_id, predicate, object_id FROM entity_relations WHERE id = ?1",
                params![result.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(subj, id_a);
        assert_eq!(pred, "works_at");
        assert_eq!(obj, id_b);
    }

    #[test]
    fn test_store_relation_dedup() {
        let mut conn = test_db();
        let id_a = insert_entity(&mut conn, "John Smith is an engineer", &embedding_a());
        let id_b = insert_entity(&mut conn, "Acme Corp is a company", &embedding_b());

        let r1 = store_relation(&conn, &id_a, "works_at", &id_b).unwrap();
        assert!(!r1.deduplicated);

        let r2 = store_relation(&conn, &id_a, "works_at", &id_b).unwrap();
        assert!(r2.deduplicated);
        assert_eq!(r2.id, r1.id);

        // Only one row in DB
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM entity_relations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_store_relation_not_entity() {
        let mut conn = test_db();
        let entity_id = insert_entity(&mut conn, "John Smith", &embedding_a());

        // Create a semantic (non-entity) memory
        let semantic_id = store::store_memory(
            &mut conn,
            "Rust is a language",
            MemoryType::Semantic,
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

        // Entity → Semantic should fail
        let result = store_relation(&conn, &entity_id, "related_to", &semantic_id);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be entity type"));

        // Semantic → Entity should fail
        let result = store_relation(&conn, &semantic_id, "related_to", &entity_id);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be entity type"));
    }

    #[test]
    fn test_store_relation_not_found() {
        let mut conn = test_db();
        let entity_id = insert_entity(&mut conn, "John Smith", &embedding_a());

        let result = store_relation(&conn, &entity_id, "works_at", "nonexistent-id");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));

        let result = store_relation(&conn, "nonexistent-id", "works_at", &entity_id);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_cascade_delete() {
        let mut conn = test_db();
        let id_a = insert_entity(&mut conn, "John Smith", &embedding_a());
        let id_b = insert_entity(&mut conn, "Acme Corp", &embedding_b());

        store_relation(&conn, &id_a, "works_at", &id_b).unwrap();

        // Verify relation exists
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM entity_relations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);

        // Delete subject entity — cascade should remove the relation
        conn.execute("DELETE FROM memories WHERE id = ?1", params![id_a])
            .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM entity_relations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }
}
