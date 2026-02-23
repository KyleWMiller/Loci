mod helpers;

use helpers::{test_db, test_embedding};
use loci::memory::forget::forget_memory;
use loci::memory::relations::store_relation;
use loci::memory::search::inspect_memory;
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};

#[test]
fn store_and_inspect_relation() {
    let mut conn = test_db();

    // Create two entity memories
    let alice_id = store_memory(
        &mut conn, "Alice is a software engineer", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    let acme_id = store_memory(
        &mut conn, "Acme Corp is a tech company", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(100), 0.92,
    ).unwrap().id;

    // Create relation
    let rel = store_relation(&conn, &alice_id, "works_at", &acme_id).unwrap();
    assert!(!rel.deduplicated);

    // Inspect should show relations
    let inspect = inspect_memory(&conn, &alice_id, true, false).unwrap();
    let relations = inspect.relations.unwrap();
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].predicate, "works_at");
}

#[test]
fn relation_dedup_is_idempotent() {
    let mut conn = test_db();

    let a = store_memory(
        &mut conn, "Entity A", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    let b = store_memory(
        &mut conn, "Entity B", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(100), 0.92,
    ).unwrap().id;

    let first = store_relation(&conn, &a, "knows", &b).unwrap();
    assert!(!first.deduplicated);

    let second = store_relation(&conn, &a, "knows", &b).unwrap();
    assert!(second.deduplicated);
}

#[test]
fn cascade_delete_removes_relations() {
    let mut conn = test_db();

    let a = store_memory(
        &mut conn, "Entity A", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(0), 0.92,
    ).unwrap().id;

    let b = store_memory(
        &mut conn, "Entity B", MemoryType::Entity, Scope::Global,
        Some("default"), 1.0, None, None, &test_embedding(100), 0.92,
    ).unwrap().id;

    store_relation(&conn, &a, "related_to", &b).unwrap();

    // Hard delete entity A
    forget_memory(&mut conn, &a, None, true).unwrap();

    // Relation should be gone (cascade)
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM entity_relations", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0, "cascade should remove relation when entity deleted");
}
