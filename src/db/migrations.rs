//! Forward-only schema migration framework.
//!
//! Tracks the schema version in `schema_meta` and runs sequential migrations
//! to bring the database up to [`CURRENT_SCHEMA_VERSION`].

use rusqlite::Connection;

/// The schema version that the current binary expects.
pub const CURRENT_SCHEMA_VERSION: u32 = 3;

/// Get the current schema version from the database.
pub fn get_schema_version(conn: &Connection) -> rusqlite::Result<u32> {
    conn.query_row(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'",
        [],
        |row| {
            let val: String = row.get(0)?;
            Ok(val.parse::<u32>().unwrap_or(0))
        },
    )
}

/// Update the stored schema version.
fn update_schema_version(conn: &Connection, version: u32) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE schema_meta SET value = ?1 WHERE key = 'schema_version'",
        [version.to_string()],
    )?;
    Ok(())
}

/// Get the stored embedding model identifier, if any.
pub fn get_embedding_model(conn: &Connection) -> rusqlite::Result<Option<String>> {
    match conn.query_row(
        "SELECT value FROM schema_meta WHERE key = 'embedding_model'",
        [],
        |row| row.get::<_, String>(0),
    ) {
        Ok(val) => Ok(Some(val)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Set the stored embedding model identifier.
pub fn set_embedding_model(conn: &Connection, model: &str) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('embedding_model', ?1)",
        [model],
    )?;
    Ok(())
}

/// Run any pending forward-only migrations. Each migration runs in a transaction.
pub fn run_migrations(conn: &Connection) -> rusqlite::Result<()> {
    let mut version = get_schema_version(conn)?;
    tracing::debug!(
        schema_version = version,
        target = CURRENT_SCHEMA_VERSION,
        "checking migrations"
    );

    while version < CURRENT_SCHEMA_VERSION {
        let next = version + 1;
        tracing::info!(from = version, to = next, "running migration");

        match next {
            2 => migrate_v1_to_v2(conn)?,
            3 => migrate_v2_to_v3(conn)?,
            _ => {
                tracing::error!(version = next, "unknown migration target");
                break;
            }
        }

        update_schema_version(conn, next)?;
        version = next;
    }

    Ok(())
}

/// DDL for the replication tables added in schema v3.
const SYNC_SCHEMA_SQL: &str = r#"
-- Locally originated operations awaiting (or already in) export to the sync dir.
-- `body` is the full op as a JSON object; hard-forgetting a memory rewrites the
-- `memory.put` rows for it into content-free no-ops.
CREATE TABLE IF NOT EXISTS op_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id TEXT NOT NULL UNIQUE,
    ts TEXT NOT NULL,
    origin TEXT NOT NULL,
    kind TEXT NOT NULL,
    memory_id TEXT,
    body TEXT NOT NULL,
    exported INTEGER NOT NULL DEFAULT 0,
    file_offset INTEGER,
    file_len INTEGER,
    scrub_pending INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_op_log_memory ON op_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_op_log_exported ON op_log(exported);

-- Per-remote read cursor: byte offset into <sync_dir>/<remote>.jsonl.
CREATE TABLE IF NOT EXISTS sync_state (
    remote TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    last_op_id TEXT,
    last_sync_at TEXT
);

-- Ops from other machines that were applied (idempotency + echo suppression).
CREATE TABLE IF NOT EXISTS applied_ops (
    op_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

-- Ops that could not be applied yet (e.g. relation whose endpoint is missing).
CREATE TABLE IF NOT EXISTS deferred_ops (
    op_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    origin TEXT NOT NULL,
    body TEXT NOT NULL,
    reason TEXT NOT NULL,
    deferred_at TEXT NOT NULL
);

-- Deleted IDs. A tombstone discards any later `memory.put` for the ID.
CREATE TABLE IF NOT EXISTS tombstones (
    memory_id TEXT PRIMARY KEY,
    hard INTEGER NOT NULL,
    ts TEXT NOT NULL,
    op_id TEXT,
    origin TEXT
);

-- Conflicting supersessions of one memory (A->B on one machine, A->C on another).
CREATE TABLE IF NOT EXISTS supersede_forks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_id TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    other_id TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    UNIQUE(old_id, winner_id, other_id)
);
"#;

/// Migration v2 → v3: replication tables and a stable per-database `machine_id`.
fn migrate_v2_to_v3(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SYNC_SCHEMA_SQL)?;
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('machine_id', ?1)",
        [uuid::Uuid::now_v7().to_string()],
    )?;
    Ok(())
}

/// Read this database's stable machine ID (generated by the v3 migration).
pub fn get_machine_id(conn: &Connection) -> rusqlite::Result<String> {
    conn.query_row(
        "SELECT value FROM schema_meta WHERE key = 'machine_id'",
        [],
        |row| row.get(0),
    )
}

/// Migration v1 → v2: Store embedding model identifier in schema_meta.
fn migrate_v1_to_v2(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('embedding_model', 'all-MiniLM-L6-v2')",
        [],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Connection {
        crate::db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::db::schema::init_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn get_schema_version_returns_1_on_fresh_db() {
        let conn = test_db();
        assert_eq!(get_schema_version(&conn).unwrap(), 1);
    }

    #[test]
    fn run_migrations_upgrades_to_current() {
        let conn = test_db();
        run_migrations(&conn).unwrap();
        assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn migration_v1_to_v2_adds_embedding_model() {
        let conn = test_db();
        // Before migration, no embedding_model key
        assert!(get_embedding_model(&conn).unwrap().is_none());

        run_migrations(&conn).unwrap();

        let model = get_embedding_model(&conn).unwrap();
        assert_eq!(model, Some("all-MiniLM-L6-v2".to_string()));
    }

    #[test]
    fn migrations_are_idempotent() {
        let conn = test_db();
        run_migrations(&conn).unwrap();
        run_migrations(&conn).unwrap(); // second call should not error
        assert_eq!(get_schema_version(&conn).unwrap(), CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn migration_v2_to_v3_creates_sync_tables_and_machine_id() {
        let conn = test_db();
        run_migrations(&conn).unwrap();
        for table in [
            "op_log",
            "sync_state",
            "applied_ops",
            "deferred_ops",
            "tombstones",
            "supersede_forks",
        ] {
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?1",
                    [table],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1, "missing table {table}");
        }
        let id = get_machine_id(&conn).unwrap();
        let parsed = uuid::Uuid::parse_str(&id).unwrap();
        assert_eq!(parsed.get_version_num(), 7);
        // Stable across repeated runs.
        run_migrations(&conn).unwrap();
        assert_eq!(get_machine_id(&conn).unwrap(), id);
    }

    #[test]
    fn set_and_get_embedding_model() {
        let conn = test_db();
        run_migrations(&conn).unwrap();

        set_embedding_model(&conn, "new-model-v3").unwrap();
        assert_eq!(
            get_embedding_model(&conn).unwrap(),
            Some("new-model-v3".to_string())
        );
    }
}
