mod helpers;

use helpers::{insert_memory, test_db, test_embedding};
use loci::memory::search::{recall_by_ids, recall_by_query, to_summary, SearchConfig, SearchFilter};
use loci::memory::types::{MemoryType, Scope};

#[test]
fn store_and_recall_by_query() {
    let mut conn = test_db();
    let emb_a = test_embedding(0);
    let emb_b = test_embedding(100);
    let emb_c = test_embedding(200);

    let id_a = insert_memory(
        &mut conn, "Deployed v2.3 on Friday", MemoryType::Episodic, Scope::Group, "project-x", 1.0, &emb_a,
    );
    let _id_b = insert_memory(
        &mut conn, "User prefers Rust over Go", MemoryType::Semantic, Scope::Global, "project-x", 1.0, &emb_b,
    );
    let _id_c = insert_memory(
        &mut conn, "How to run the deploy pipeline", MemoryType::Procedural, Scope::Global, "project-x", 1.0, &emb_c,
    );

    // Query with emb_a should return the episodic memory first
    let filter = SearchFilter {
        memory_type: None,
        scope: None,
        group: "project-x".to_string(),
        min_confidence: 0.0,
    };
    let config = SearchConfig {
        max_results: 10,
        token_budget: 10000,
        rrf_k: 60,
    };

    let response = recall_by_query(&conn, &emb_a, "deployed friday", &filter, &config).unwrap();
    assert!(!response.results.is_empty(), "should return at least one result");
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&id_a.as_str()), "episodic memory should be in results");
}

#[test]
fn recall_by_ids_hydrates_memories() {
    let mut conn = test_db();
    let emb = test_embedding(10);

    let id = insert_memory(
        &mut conn, "Some important fact", MemoryType::Semantic, Scope::Global, "default", 0.8, &emb,
    );

    let response = recall_by_ids(&conn, &[id.clone()]).unwrap();
    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].id, id);
    assert_eq!(response.results[0].content, "Some important fact");
    assert_eq!(response.results[0].confidence, 0.8);
}

#[test]
fn recall_with_type_filter() {
    let mut conn = test_db();

    insert_memory(
        &mut conn, "An episodic memory", MemoryType::Episodic, Scope::Group, "default", 1.0, &test_embedding(0),
    );
    insert_memory(
        &mut conn, "A semantic memory", MemoryType::Semantic, Scope::Global, "default", 1.0, &test_embedding(100),
    );

    let filter = SearchFilter {
        memory_type: Some(MemoryType::Semantic),
        scope: None,
        group: "default".to_string(),
        min_confidence: 0.0,
    };
    let config = SearchConfig {
        max_results: 10,
        token_budget: 10000,
        rrf_k: 60,
    };

    let response = recall_by_query(&conn, &test_embedding(100), "semantic", &filter, &config).unwrap();
    // All results should be semantic type
    for r in &response.results {
        assert_eq!(r.memory_type, "semantic");
    }
}

#[test]
fn summary_only_truncates_content() {
    let mut conn = test_db();
    let long_content = "A".repeat(200);
    let emb = test_embedding(5);

    insert_memory(
        &mut conn, &long_content, MemoryType::Semantic, Scope::Global, "default", 1.0, &emb,
    );

    let filter = SearchFilter {
        memory_type: None,
        scope: None,
        group: "default".to_string(),
        min_confidence: 0.0,
    };
    let config = SearchConfig {
        max_results: 10,
        token_budget: 10000,
        rrf_k: 60,
    };

    let response = recall_by_query(&conn, &emb, "test", &filter, &config).unwrap();
    let summary = to_summary(&response);
    assert!(!summary.results.is_empty());
    // Preview should be truncated (80 chars + "...")
    assert!(summary.results[0].preview.len() <= 83);
}
