use rusqlite::Connection;

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

/// Run any pending migrations. Currently a stub — the initial schema is version 1.
/// Future milestones will add migration functions here.
pub fn run_migrations(conn: &Connection) -> rusqlite::Result<()> {
    let version = get_schema_version(conn)?;
    tracing::debug!(schema_version = version, "current schema version");
    // No migrations yet — schema is at version 1
    Ok(())
}
