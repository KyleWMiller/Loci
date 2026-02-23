mod helpers;

use helpers::{insert_memory, test_db, test_embedding};
use loci::memory::forget::forget_memory;
use loci::memory::search::{inspect_memory, recall_by_ids};
use loci::memory::types::{MemoryType, Scope};

#[test]
fn soft_delete_marks_as_forgotten() {
    let mut conn = test_db();
    let id = insert_memory(
        &mut conn, "Something to forget", MemoryType::Episodic, Scope::Group, "default", 1.0, &test_embedding(0),
    );

    let result = forget_memory(&mut conn, &id, Some("no longer relevant"), false).unwrap();
    assert!(!result.hard_deleted);

    // Inspect should show superseded_by = "forgotten"
    let inspect = inspect_memory(&conn, &id, false, false).unwrap();
    assert_eq!(inspect.memory.superseded_by.as_deref(), Some("forgotten"));
}

#[test]
fn hard_delete_removes_completely() {
    let mut conn = test_db();
    let id = insert_memory(
        &mut conn, "Something to permanently delete", MemoryType::Semantic, Scope::Global, "default", 1.0, &test_embedding(10),
    );

    let result = forget_memory(&mut conn, &id, None, true).unwrap();
    assert!(result.hard_deleted);

    // Memory should be gone from memories table
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE id = ?1",
            [&id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 0, "hard delete should remove from memories table");

    // Should be gone from vec table
    let vec_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_vec WHERE id = ?1",
            [&id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(vec_count, 0, "hard delete should remove from vec table");

    // recall_by_ids should return empty
    let response = recall_by_ids(&conn, &[id]).unwrap();
    assert!(response.results.is_empty());
}
