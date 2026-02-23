pub mod migrations;
pub mod schema;

use anyhow::{Context, Result};
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;
use std::path::Path;
use std::sync::Once;

static SQLITE_VEC_INIT: Once = Once::new();

/// Register the sqlite-vec extension globally. Safe to call multiple times.
pub fn load_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    });
}

/// Open (or create) the Loci database at the given path, with all extensions
/// loaded and schema initialized.
pub fn open_database(path: impl AsRef<Path>) -> Result<Connection> {
    let path = path.as_ref();

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    load_sqlite_vec();

    let conn = Connection::open(path).with_context(|| {
        format!(
            "failed to open database at {}. If the file is corrupt, \
             restore from a backup or run `loci reset` to start fresh.",
            path.display()
        )
    })?;

    // Enable WAL mode for better concurrent read performance
    conn.pragma_update(None, "journal_mode", "WAL")?;
    // Enable foreign keys
    conn.pragma_update(None, "foreign_keys", "ON")?;
    // Wait up to 5 seconds for locks instead of failing immediately
    conn.pragma_update(None, "busy_timeout", "5000")?;

    schema::init_schema(&conn).context("failed to initialize schema")?;
    migrations::run_migrations(&conn).context("failed to run migrations")?;

    // Quick integrity check after schema init
    let integrity: String = conn.pragma_query_value(None, "quick_check", |row| row.get(0))?;
    if integrity != "ok" {
        anyhow::bail!(
            "database integrity check failed: {integrity}. \
             Try restoring from a backup (`loci export` from a good copy, \
             then `loci reset && loci import backup.json`)."
        );
    }

    tracing::info!(path = %path.display(), "database initialized");
    Ok(conn)
}

/// Result of a full database health check.
pub struct HealthReport {
    pub schema_version: u32,
    pub embedding_model: Option<String>,
    pub integrity_ok: bool,
    pub integrity_details: String,
    pub sqlite_vec_version: String,
    pub memory_count: i64,
    pub relation_count: i64,
    pub log_count: i64,
}

/// Run a comprehensive health check on the database.
pub fn check_database_health(conn: &Connection) -> Result<HealthReport> {
    let schema_version = migrations::get_schema_version(conn)
        .context("failed to read schema version")?;

    let embedding_model = migrations::get_embedding_model(conn)
        .context("failed to read embedding model")?;

    let integrity_details: String = conn
        .pragma_query_value(None, "integrity_check", |row| row.get(0))
        .context("failed to run integrity check")?;
    let integrity_ok = integrity_details == "ok";

    let sqlite_vec_version: String = conn
        .query_row("SELECT vec_version()", [], |row| row.get(0))
        .context("failed to get sqlite-vec version")?;

    let memory_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap_or(0);

    let relation_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM entity_relations", [], |row| row.get(0))
        .unwrap_or(0);

    let log_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_log", [], |row| row.get(0))
        .unwrap_or(0);

    Ok(HealthReport {
        schema_version,
        embedding_model,
        integrity_ok,
        integrity_details,
        sqlite_vec_version,
        memory_count,
        relation_count,
        log_count,
    })
}

/// Open an in-memory database for testing.
#[cfg(test)]
pub fn open_memory_database() -> Result<Connection> {
    load_sqlite_vec();
    let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    schema::init_schema(&conn).context("failed to initialize schema")?;
    migrations::run_migrations(&conn).context("failed to run migrations")?;
    Ok(conn)
}
