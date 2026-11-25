mod helpers;

use loci::db;
use loci::db::migrations::{
    CURRENT_SCHEMA_VERSION, get_embedding_model, get_schema_version, run_migrations,
};

#[test]
fn fresh_db_migrates_to_current_version() {
    let conn = helpers::test_db();
    assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
}

#[test]
fn migration_adds_embedding_model_key() {
    let conn = helpers::test_db();
    let model = get_embedding_model(&conn).unwrap();
    assert_eq!(model, Some("all-MiniLM-L6-v2".to_string()));
}

#[test]
fn migrations_are_idempotent() {
    let conn = helpers::test_db();
    // Running again should be a no-op
    run_migrations(&conn).unwrap();
    assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
}

#[test]
fn manual_v1_db_upgrades_correctly() {
    // Simulate a v1 database that hasn't been migrated
    db::load_sqlite_vec();
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    db::schema::init_schema(&conn).unwrap();

    // Verify it starts at v1
    assert_eq!(get_schema_version(&conn).unwrap(), 1);
    assert!(get_embedding_model(&conn).unwrap().is_none());

    // Run migrations
    run_migrations(&conn).unwrap();

    // Should now be at current version
    assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
    assert!(get_embedding_model(&conn).unwrap().is_some());
}

/// The exact DDL shipped by loci-mcp 0.2.3 (schema version 2), with synthetic rows.
const V023_SCHEMA: &str = r#"
CREATE TABLE memories (
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
CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_memories_scope ON memories(scope);
CREATE INDEX idx_memories_group ON memories(source_group);
CREATE INDEX idx_memories_confidence ON memories(confidence);
CREATE INDEX idx_memories_superseded ON memories(superseded_by);
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, id UNINDEXED, type UNINDEXED, content='memories', content_rowid='rowid'
);
CREATE TABLE entity_relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE TABLE memory_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL CHECK(operation IN ('create','update','supersede','decay','compact','delete')),
    memory_id TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIRTUAL TABLE memories_vec USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[384]);
INSERT INTO schema_meta VALUES ('schema_version', '2');
INSERT INTO schema_meta VALUES ('embedding_model', 'all-MiniLM-L6-v2');
INSERT INTO memories (id, type, content, created_at, updated_at, superseded_by)
    VALUES ('m-old', 'semantic', 'synthetic old fact', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'm-new');
INSERT INTO memories (id, type, content, created_at, updated_at)
    VALUES ('m-new', 'semantic', 'synthetic new fact', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z');
INSERT INTO memory_log (operation, memory_id, details, created_at)
    VALUES ('create', 'm-new', NULL, '2026-01-02T00:00:00Z');
"#;

#[test]
fn v0_2_3_database_migrates_in_place() {
    db::load_sqlite_vec();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(V023_SCHEMA).unwrap();
        assert_eq!(get_schema_version(&conn).unwrap(), 2);
    }

    // The real open path: init_schema (IF NOT EXISTS) then migrations.
    let conn = db::open_database(&path).unwrap();
    assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
    assert_eq!(CURRENT_SCHEMA_VERSION, 4);

    // Existing data untouched.
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 2);
    let sb: String = conn
        .query_row(
            "SELECT superseded_by FROM memories WHERE id = 'm-old'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(sb, "m-new");
    let logs: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_log", [], |r| r.get(0))
        .unwrap();
    assert_eq!(logs, 1);
    assert_eq!(
        get_embedding_model(&conn).unwrap(),
        Some("all-MiniLM-L6-v2".to_string())
    );

    // New machinery present, machine_id stable across reopen.
    let id = loci::db::migrations::get_machine_id(&conn).unwrap();
    assert_eq!(uuid::Uuid::parse_str(&id).unwrap().get_version_num(), 7);
    for t in [
        "op_log",
        "sync_state",
        "tombstones",
        "supersede_forks",
        "supersessions",
    ] {
        conn.query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();
    }
    drop(conn);
    let conn = db::open_database(&path).unwrap();
    assert_eq!(loci::db::migrations::get_machine_id(&conn).unwrap(), id);
}
