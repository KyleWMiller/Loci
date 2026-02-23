mod helpers;

use helpers::{similar_embedding, test_db, test_embedding};
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};

#[test]
fn dedup_merges_similar_same_type() {
    let mut conn = test_db();
    let emb_a = test_embedding(0);
    let emb_b = similar_embedding(&emb_a);

    let result_a = store_memory(
        &mut conn,
        "User prefers dark mode",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        None,
        &emb_a,
        0.92,
    )
    .unwrap();
    assert!(!result_a.deduplicated);

    let result_b = store_memory(
        &mut conn,
        "User prefers dark theme",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        None,
        &emb_b,
        0.92,
    )
    .unwrap();

    // Should be deduplicated — same type, high cosine similarity
    assert!(result_b.deduplicated);
    assert_eq!(result_b.id, result_a.id, "should return the existing memory ID");
}

#[test]
fn dedup_does_not_merge_different_types() {
    let mut conn = test_db();
    let emb_a = test_embedding(0);
    let emb_b = similar_embedding(&emb_a);

    let result_a = store_memory(
        &mut conn,
        "Deployment happened on Friday",
        MemoryType::Episodic,
        Scope::Group,
        Some("default"),
        1.0,
        None,
        None,
        &emb_a,
        0.92,
    )
    .unwrap();
    assert!(!result_a.deduplicated);

    let result_b = store_memory(
        &mut conn,
        "Deployments happen on Fridays",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        None,
        &emb_b,
        0.92,
    )
    .unwrap();

    // Different types should NOT be deduplicated
    assert!(!result_b.deduplicated);
    assert_ne!(result_b.id, result_a.id);
}

#[test]
fn dedup_does_not_merge_distant_embeddings() {
    let mut conn = test_db();
    let emb_a = test_embedding(0);
    let emb_b = test_embedding(200); // very different

    let result_a = store_memory(
        &mut conn,
        "A memory about dogs",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        None,
        &emb_a,
        0.92,
    )
    .unwrap();

    let result_b = store_memory(
        &mut conn,
        "A memory about cats",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        None,
        &emb_b,
        0.92,
    )
    .unwrap();

    assert!(!result_b.deduplicated);
    assert_ne!(result_b.id, result_a.id);
}
