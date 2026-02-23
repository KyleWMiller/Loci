mod helpers;

use helpers::{test_db, test_embedding};
use loci::memory::search::{recall_by_query, SearchConfig, SearchFilter};
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};

#[test]
fn superseded_memory_excluded_from_search() {
    let mut conn = test_db();
    let emb_a = test_embedding(0);
    let emb_b = test_embedding(100);

    // Store original
    let result_a = store_memory(
        &mut conn,
        "User prefers npm",
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

    // Store replacement that supersedes original
    let result_b = store_memory(
        &mut conn,
        "User prefers bun over npm",
        MemoryType::Semantic,
        Scope::Global,
        Some("default"),
        1.0,
        None,
        Some(&result_a.id),
        &emb_b,
        0.92,
    )
    .unwrap();

    assert!(!result_b.deduplicated);
    assert_eq!(result_b.superseded, Some(result_a.id.clone()));

    // Search — should find B but not A
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

    let response = recall_by_query(&conn, &emb_a, "user prefers", &filter, &config).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(!ids.contains(&result_a.id.as_str()), "superseded memory should be excluded");
    assert!(ids.contains(&result_b.id.as_str()), "replacement should be found");
}
