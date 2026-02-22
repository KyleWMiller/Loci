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

    let conn = Connection::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    // Enable WAL mode for better concurrent read performance
    conn.pragma_update(None, "journal_mode", "WAL")?;
    // Enable foreign keys
    conn.pragma_update(None, "foreign_keys", "ON")?;

    schema::init_schema(&conn).context("failed to initialize schema")?;
    migrations::run_migrations(&conn).context("failed to run migrations")?;

    tracing::info!(path = %path.display(), "database initialized");
    Ok(conn)
}

/// Open an in-memory database for testing.
#[cfg(test)]
pub fn open_memory_database() -> Result<Connection> {
    load_sqlite_vec();
    let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    schema::init_schema(&conn).context("failed to initialize schema")?;
    Ok(conn)
}
