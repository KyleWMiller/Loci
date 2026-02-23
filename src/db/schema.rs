//! SQL DDL for all Loci tables.
//!
//! Defines the `memories`, `memories_fts` (FTS5), `memories_vec` (vec0),
//! `entity_relations`, `memory_log`, and `schema_meta` tables. All DDL uses
//! `IF NOT EXISTS` for idempotent initialization.

use rusqlite::Connection;

/// All schema DDL statements for Loci's core tables.
const SCHEMA_SQL: &str = r#"
-- Core memory storage
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('episodic','semantic','procedural','entity')),
    content TEXT NOT NULL,
    source_group TEXT,
    scope TEXT NOT NULL DEFAULT 'global' CHECK(scope IN ('global','group')),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    superseded_by TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_group ON memories(source_group);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by);

-- Full-text search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    id UNINDEXED,
    type UNINDEXED,
    content='memories',
    content_rowid='rowid'
);

-- Entity relationship graph
CREATE TABLE IF NOT EXISTS entity_relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relations_subject ON entity_relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON entity_relations(object_id);
CREATE INDEX IF NOT EXISTS idx_relations_predicate ON entity_relations(predicate);

-- Audit log
CREATE TABLE IF NOT EXISTS memory_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL CHECK(operation IN ('create','update','supersede','decay','compact','delete')),
    memory_id TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);

-- Schema metadata
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// vec0 virtual table must be created separately (sqlite-vec syntax).
const VEC_TABLE_SQL: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[384]
);
"#;

/// Initialize all schema tables. Idempotent (uses IF NOT EXISTS).
pub fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SCHEMA_SQL)?;
    conn.execute_batch(VEC_TABLE_SQL)?;

    // Set initial schema version if not already present
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1')",
        [],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_creates_all_tables() {
        crate::db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Verify all tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(tables.contains(&"memories".to_string()));
        assert!(tables.contains(&"entity_relations".to_string()));
        assert!(tables.contains(&"memory_log".to_string()));
        assert!(tables.contains(&"schema_meta".to_string()));

        // Verify virtual tables exist
        let version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .unwrap();
        assert!(!version.is_empty());
    }

    #[test]
    fn schema_is_idempotent() {
        crate::db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap(); // second call should not error
    }
}
