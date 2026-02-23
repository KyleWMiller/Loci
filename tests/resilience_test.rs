use loci::db;
use tempfile::TempDir;

#[test]
fn open_creates_new_db_at_nonexistent_path() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("subdir").join("new.db");

    // Should not exist yet
    assert!(!db_path.exists());

    let conn = db::open_database(&db_path).unwrap();

    // Should have been created
    assert!(db_path.exists());

    // Should be functional
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn health_check_passes_on_valid_db() {
    db::load_sqlite_vec();
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    db::schema::init_schema(&conn).unwrap();
    db::migrations::run_migrations(&conn).unwrap();

    let report = db::check_database_health(&conn).unwrap();
    assert!(report.integrity_ok);
    assert_eq!(report.schema_version, db::migrations::CURRENT_SCHEMA_VERSION);
    assert!(!report.sqlite_vec_version.is_empty());
    assert_eq!(report.memory_count, 0);
    assert_eq!(report.relation_count, 0);
    assert_eq!(report.log_count, 0);
}

#[test]
fn busy_timeout_is_set() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");

    let conn = db::open_database(&db_path).unwrap();

    let timeout: i64 = conn
        .pragma_query_value(None, "busy_timeout", |row| row.get(0))
        .unwrap();
    assert_eq!(timeout, 5000);
}
