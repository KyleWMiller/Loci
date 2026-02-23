mod helpers;

use loci::db;
use loci::db::migrations::{get_schema_version, run_migrations, get_embedding_model, CURRENT_SCHEMA_VERSION};

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
