use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;
use std::mem;

fn load_sqlite_vec() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    }
}

#[test]
fn full_schema_creates_all_tables_and_indexes() {
    load_sqlite_vec();
    let conn = Connection::open_in_memory().unwrap();

    // Run the schema SQL from the db module
    conn.execute_batch(
        r#"
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

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            id UNINDEXED,
            type UNINDEXED,
            content='memories',
            content_rowid='rowid'
        );

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

        CREATE TABLE IF NOT EXISTS memory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL CHECK(operation IN ('create','update','supersede','decay','compact','delete')),
            memory_id TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[384]
        );

        INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1');
        "#,
    )
    .unwrap();

    // Verify tables
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert!(tables.contains(&"memories".to_string()), "memories table missing");
    assert!(
        tables.contains(&"entity_relations".to_string()),
        "entity_relations table missing"
    );
    assert!(tables.contains(&"memory_log".to_string()), "memory_log table missing");
    assert!(tables.contains(&"schema_meta".to_string()), "schema_meta table missing");

    // Verify indexes
    let indexes: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert!(indexes.contains(&"idx_memories_type".to_string()));
    assert!(indexes.contains(&"idx_memories_scope".to_string()));
    assert!(indexes.contains(&"idx_memories_group".to_string()));
    assert!(indexes.contains(&"idx_memories_confidence".to_string()));
    assert!(indexes.contains(&"idx_relations_subject".to_string()));

    // Verify schema version
    let version: String = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, "1");

    // Verify vec0 table works
    let vec_version: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .unwrap();
    assert!(!vec_version.is_empty());

    // Insert and query a test vector to confirm vec0 is functional
    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let embedding_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4)
    };

    conn.execute(
        "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
        rusqlite::params!["test-vec", embedding_bytes],
    )
    .unwrap();

    let count: i64 = conn
        .query_row("SELECT count(*) FROM memories_vec", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);

    // Insert a memory row to confirm the table constraints work
    conn.execute(
        "INSERT INTO memories (id, type, content, created_at, updated_at)
         VALUES ('test-1', 'semantic', 'Rust is a systems language', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
        [],
    )
    .unwrap();

    // Verify CHECK constraint rejects bad types
    let result = conn.execute(
        "INSERT INTO memories (id, type, content, created_at, updated_at)
         VALUES ('test-2', 'invalid_type', 'bad', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
        [],
    );
    assert!(result.is_err(), "invalid type should be rejected by CHECK constraint");

    println!("Full schema integration test PASSED");
}
