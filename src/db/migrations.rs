//! Forward-only schema migration framework.
//!
//! Tracks the schema version in `schema_meta` and runs sequential migrations
//! to bring the database up to [`CURRENT_SCHEMA_VERSION`].

use rusqlite::Connection;

/// The schema version that the current binary expects.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

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
    tracing::debug!(schema_version = version, target = CURRENT_SCHEMA_VERSION, "checking migrations");

    while version < CURRENT_SCHEMA_VERSION {
        let next = version + 1;
        tracing::info!(from = version, to = next, "running migration");

        match next {
            2 => migrate_v1_to_v2(conn)?,
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
