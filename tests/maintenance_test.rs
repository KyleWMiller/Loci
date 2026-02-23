mod helpers;

use helpers::{test_db, test_embedding};
use loci::config::MaintenanceConfig;
use loci::memory::maintenance::{apply_decay, cleanup_stale};
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};
use rusqlite::params;

/// Backdate a memory's created_at and last_accessed to simulate aging.
fn backdate_memory(conn: &rusqlite::Connection, id: &str, days_ago: i64) {
    let old_date = (chrono::Utc::now() - chrono::Duration::days(days_ago)).to_rfc3339();
    conn.execute(
        "UPDATE memories SET created_at = ?1, updated_at = ?1, last_accessed = ?1 WHERE id = ?2",
        params![old_date, id],
    )
    .unwrap();
}

#[test]
fn decay_reduces_confidence() {
    let mut conn = test_db();
    let config = MaintenanceConfig::default();

    let id = store_memory(
        &mut conn, "Old event", MemoryType::Episodic, Scope::Group,
        Some("default"), 1.0, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    let result = apply_decay(&conn, &config).unwrap();
    let total: usize = result.affected_by_type.values().sum();
    assert!(total > 0, "should have decayed at least one memory");

    let confidence: f64 = conn
        .query_row("SELECT confidence FROM memories WHERE id = ?1", [&id], |row| row.get(0))
        .unwrap();
    assert!(confidence < 1.0, "confidence should have decreased from 1.0");
    assert_eq!(confidence, config.episodic_decay_factor, "should match episodic decay factor");
}

#[test]
fn cleanup_stale_removes_low_confidence_old_memories() {
    let mut conn = test_db();
    let mut config = MaintenanceConfig::default();
    config.cleanup_confidence_floor = 0.1;
    config.cleanup_no_access_days = 30;

    let id = store_memory(
        &mut conn, "Very old and unimportant", MemoryType::Episodic, Scope::Group,
        Some("default"), 0.05, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    // Backdate so it's stale
    backdate_memory(&conn, &id, 60);

    // Dry run first
    let dry = cleanup_stale(&mut conn, &config, true).unwrap();
    assert_eq!(dry.candidates.len(), 1, "should find one stale candidate");
    assert_eq!(dry.deleted, 0, "dry run should not delete");

    // Now actually cleanup
    let result = cleanup_stale(&mut conn, &config, false).unwrap();
    assert_eq!(result.deleted, 1, "should hard-delete one stale memory");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [&id], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0, "memory should be gone after cleanup");
}

#[test]
fn cleanup_skips_high_confidence_memories() {
    let mut conn = test_db();
    let mut config = MaintenanceConfig::default();
    config.cleanup_confidence_floor = 0.1;
    config.cleanup_no_access_days = 30;

    let id = store_memory(
        &mut conn, "Important memory", MemoryType::Semantic, Scope::Global,
        Some("default"), 0.5, None, None, &test_embedding(10), 0.92,
    ).unwrap().id;

    backdate_memory(&conn, &id, 60);

    let result = cleanup_stale(&mut conn, &config, false).unwrap();
    assert_eq!(result.deleted, 0, "should not delete memory with confidence above floor");
}

#[test]
fn decay_skips_superseded_memories() {
    let mut conn = test_db();
    let config = MaintenanceConfig::default();

    let id_a = store_memory(
        &mut conn, "Old version", MemoryType::Semantic, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    // Supersede it
    store_memory(
        &mut conn, "New version", MemoryType::Semantic, Scope::Global,
        Some("default"), 1.0, None, Some(&id_a), &test_embedding(100), 0.92,
    ).unwrap();

    let before: f64 = conn
        .query_row("SELECT confidence FROM memories WHERE id = ?1", [&id_a], |row| row.get(0))
        .unwrap();

    apply_decay(&conn, &config).unwrap();

    let after: f64 = conn
        .query_row("SELECT confidence FROM memories WHERE id = ?1", [&id_a], |row| row.get(0))
        .unwrap();

    assert_eq!(before, after, "superseded memory confidence should not decay");
}
