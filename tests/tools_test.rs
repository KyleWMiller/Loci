//! Integration tests for MCP tool handlers.
//!
//! Tests the `LociTools` async handlers end-to-end: parameter validation,
//! store → recall → inspect → forget workflows, and edge cases.

mod helpers;

use loci::config::LociConfig;
use loci::db;
use loci::embedding::EmbeddingProvider;
use loci::tools::LociTools;
use rmcp::handler::server::wrapper::Parameters;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

/// Deterministic embedding provider for tests — returns a spike vector at `hash(text) % 384`.
struct FakeEmbedding;

impl EmbeddingProvider for FakeEmbedding {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut v = vec![0.0f32; 384];
        // Simple hash: sum of bytes mod 384
        let idx = text.bytes().map(|b| b as usize).sum::<usize>() % 384;
        v[idx] = 1.0;
        Ok(v)
    }
}

/// Create a `LociTools` instance backed by an in-memory DB.
fn test_tools() -> LociTools {
    db::load_sqlite_vec();
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    db::schema::init_schema(&conn).unwrap();
    db::migrations::run_migrations(&conn).unwrap();

    let db = Arc::new(Mutex::new(conn));
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(FakeEmbedding);
    let config = Arc::new(LociConfig::default());

    LociTools::new(db, embedding, config)
}

// ── store_memory ────────────────────────────────────────────────────────

#[tokio::test]
async fn store_memory_happy_path() {
    let tools = test_tools();
    let params = loci::tools::store_memory::StoreMemoryParams {
        content: "Rust is a systems programming language".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };

    let result = tools.store_memory(Parameters(params)).await;
    assert!(result.is_ok(), "store_memory failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert!(json["id"].is_string());
    assert_eq!(json["deduplicated"], false);
}

#[tokio::test]
async fn store_memory_empty_content_rejected() {
    let tools = test_tools();
    let params = loci::tools::store_memory::StoreMemoryParams {
        content: "".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };

    let result = tools.store_memory(Parameters(params)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("content must not be empty"));
}

#[tokio::test]
async fn store_memory_invalid_type_rejected() {
    let tools = test_tools();
    let params = loci::tools::store_memory::StoreMemoryParams {
        content: "some content".into(),
        r#type: "invalid_type".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };

    let result = tools.store_memory(Parameters(params)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn store_memory_confidence_out_of_range_rejected() {
    let tools = test_tools();
    let params = loci::tools::store_memory::StoreMemoryParams {
        content: "some content".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: Some(1.5),
        metadata: None,
        supersedes: None,
    };

    let result = tools.store_memory(Parameters(params)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("confidence must be between"));
}

#[tokio::test]
async fn store_memory_with_metadata() {
    let tools = test_tools();
    let params = loci::tools::store_memory::StoreMemoryParams {
        content: "meeting notes from standup".into(),
        r#type: "episodic".into(),
        group: Some("project-x".into()),
        scope: Some("group".into()),
        confidence: Some(0.9),
        metadata: Some(serde_json::json!({"attendees": ["alice", "bob"]})),
        supersedes: None,
    };

    let result = tools.store_memory(Parameters(params)).await;
    assert!(result.is_ok(), "store_memory failed: {:?}", result.err());
}

// ── recall_memory ───────────────────────────────────────────────────────

#[tokio::test]
async fn recall_memory_missing_query_and_ids_rejected() {
    let tools = test_tools();
    let params = loci::tools::recall_memory::RecallMemoryParams {
        query: None,
        ids: None,
        r#type: None,
        scope: None,
        group: None,
        max_results: None,
        summary_only: None,
        token_budget: None,
        min_confidence: None,
    };

    let result = tools.recall_memory(Parameters(params)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("either 'query' or 'ids'"));
}

#[tokio::test]
async fn recall_memory_by_query() {
    let tools = test_tools();

    // Store a memory first
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "Loci uses SQLite for storage".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    tools.store_memory(Parameters(store_params)).await.unwrap();

    // Recall it
    let recall_params = loci::tools::recall_memory::RecallMemoryParams {
        query: Some("SQLite storage".into()),
        ids: None,
        r#type: None,
        scope: None,
        group: None,
        max_results: Some(5),
        summary_only: None,
        token_budget: None,
        min_confidence: None,
    };

    let result = tools.recall_memory(Parameters(recall_params)).await;
    assert!(result.is_ok(), "recall_memory failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert!(json["results"].is_array());
}

#[tokio::test]
async fn recall_memory_by_ids() {
    let tools = test_tools();

    // Store a memory
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "Test memory for ID recall".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    let stored: serde_json::Value =
        serde_json::from_str(&tools.store_memory(Parameters(store_params)).await.unwrap())
            .unwrap();
    let id = stored["id"].as_str().unwrap().to_string();

    // Recall by ID
    let recall_params = loci::tools::recall_memory::RecallMemoryParams {
        query: None,
        ids: Some(vec![id.clone()]),
        r#type: None,
        scope: None,
        group: None,
        max_results: None,
        summary_only: None,
        token_budget: None,
        min_confidence: None,
    };

    let result = tools.recall_memory(Parameters(recall_params)).await;
    assert!(result.is_ok(), "recall by ID failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"].as_str().unwrap(), id);
}

#[tokio::test]
async fn recall_memory_summary_only() {
    let tools = test_tools();

    // Store
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "A sufficiently long memory content that should be truncated in summary mode to verify truncation works correctly".into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    tools.store_memory(Parameters(store_params)).await.unwrap();

    // Recall with summary_only
    let recall_params = loci::tools::recall_memory::RecallMemoryParams {
        query: Some("long memory".into()),
        ids: None,
        r#type: None,
        scope: None,
        group: None,
        max_results: None,
        summary_only: Some(true),
        token_budget: None,
        min_confidence: None,
    };

    let result = tools.recall_memory(Parameters(recall_params)).await;
    assert!(result.is_ok());
}

// ── forget_memory ───────────────────────────────────────────────────────

#[tokio::test]
async fn forget_memory_empty_id_rejected() {
    let tools = test_tools();
    let params = loci::tools::forget_memory::ForgetMemoryParams {
        memory_id: "".into(),
        reason: None,
        hard_delete: None,
    };

    let result = tools.forget_memory(Parameters(params)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("memory_id must not be empty"));
}

#[tokio::test]
async fn forget_memory_nonexistent_id_fails() {
    let tools = test_tools();
    let params = loci::tools::forget_memory::ForgetMemoryParams {
        memory_id: "nonexistent-id".into(),
        reason: Some("testing".into()),
        hard_delete: None,
    };

    let result = tools.forget_memory(Parameters(params)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn forget_memory_soft_delete() {
    let tools = test_tools();

    // Store then forget
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "Memory to soft delete".into(),
        r#type: "episodic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    let stored: serde_json::Value =
        serde_json::from_str(&tools.store_memory(Parameters(store_params)).await.unwrap())
            .unwrap();
    let id = stored["id"].as_str().unwrap().to_string();

    let forget_params = loci::tools::forget_memory::ForgetMemoryParams {
        memory_id: id.clone(),
        reason: Some("no longer relevant".into()),
        hard_delete: None,
    };

    let result = tools.forget_memory(Parameters(forget_params)).await;
    assert!(result.is_ok(), "forget failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["id"].as_str().unwrap(), id);
    assert_eq!(json["hard_deleted"], false);
}

#[tokio::test]
async fn forget_memory_hard_delete() {
    let tools = test_tools();

    // Store then hard delete
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "Memory to hard delete".into(),
        r#type: "episodic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    let stored: serde_json::Value =
        serde_json::from_str(&tools.store_memory(Parameters(store_params)).await.unwrap())
            .unwrap();
    let id = stored["id"].as_str().unwrap().to_string();

    let forget_params = loci::tools::forget_memory::ForgetMemoryParams {
        memory_id: id.clone(),
        reason: None,
        hard_delete: Some(true),
    };

    let result = tools.forget_memory(Parameters(forget_params)).await;
    assert!(result.is_ok());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["hard_deleted"], true);
}

// ── memory_stats ────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_stats_empty_db() {
    let tools = test_tools();
    let params = loci::tools::memory_stats::MemoryStatsParams { group: None };

    let result = tools.memory_stats(Parameters(params)).await;
    assert!(result.is_ok(), "stats failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["total_memories"], 0);
}

#[tokio::test]
async fn memory_stats_after_store() {
    let tools = test_tools();

    // Store two memories
    for content in ["fact one", "fact two"] {
        let params = loci::tools::store_memory::StoreMemoryParams {
            content: content.into(),
            r#type: "semantic".into(),
            group: None,
            scope: None,
            confidence: None,
            metadata: None,
            supersedes: None,
        };
        tools.store_memory(Parameters(params)).await.unwrap();
    }

    let params = loci::tools::memory_stats::MemoryStatsParams { group: None };
    let result = tools.memory_stats(Parameters(params)).await;
    assert!(result.is_ok());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["total_memories"], 2);
}

// ── memory_inspect ──────────────────────────────────────────────────────

#[tokio::test]
async fn memory_inspect_happy_path() {
    let tools = test_tools();

    // Store
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "Memory to inspect".into(),
        r#type: "entity".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: None,
        supersedes: None,
    };
    let stored: serde_json::Value =
        serde_json::from_str(&tools.store_memory(Parameters(store_params)).await.unwrap())
            .unwrap();
    let id = stored["id"].as_str().unwrap().to_string();

    // Inspect with relations and log
    let params = loci::tools::memory_inspect::MemoryInspectParams {
        memory_id: id.clone(),
        include_relations: Some(true),
        include_log: Some(true),
    };

    let result = tools.memory_inspect(Parameters(params)).await;
    assert!(result.is_ok(), "inspect failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(json["memory"]["id"].as_str().unwrap(), id);
    assert_eq!(json["memory"]["type"].as_str().unwrap(), "entity");
    assert!(json["log"].is_array());
}

#[tokio::test]
async fn memory_inspect_nonexistent_fails() {
    let tools = test_tools();
    let params = loci::tools::memory_inspect::MemoryInspectParams {
        memory_id: "does-not-exist".into(),
        include_relations: None,
        include_log: None,
    };

    let result = tools.memory_inspect(Parameters(params)).await;
    assert!(result.is_err());
}

// ── store_relation ──────────────────────────────────────────────────────

#[tokio::test]
async fn store_relation_empty_fields_rejected() {
    let tools = test_tools();

    // Empty subject_id
    let params = loci::tools::store_relation::StoreRelationParams {
        subject_id: "".into(),
        predicate: "works_at".into(),
        object_id: "some-id".into(),
    };
    assert!(tools.store_relation(Parameters(params)).await.is_err());

    // Empty predicate
    let params = loci::tools::store_relation::StoreRelationParams {
        subject_id: "some-id".into(),
        predicate: "".into(),
        object_id: "some-id".into(),
    };
    assert!(tools.store_relation(Parameters(params)).await.is_err());

    // Empty object_id
    let params = loci::tools::store_relation::StoreRelationParams {
        subject_id: "some-id".into(),
        predicate: "works_at".into(),
        object_id: "".into(),
    };
    assert!(tools.store_relation(Parameters(params)).await.is_err());
}

#[tokio::test]
async fn store_relation_between_entities() {
    let tools = test_tools();

    // Create two entity memories
    let mut ids = Vec::new();
    for name in ["Alice", "Acme Corp"] {
        let params = loci::tools::store_memory::StoreMemoryParams {
            content: name.into(),
            r#type: "entity".into(),
            group: None,
            scope: None,
            confidence: None,
            metadata: None,
            supersedes: None,
        };
        let stored: serde_json::Value =
            serde_json::from_str(&tools.store_memory(Parameters(params)).await.unwrap())
                .unwrap();
        ids.push(stored["id"].as_str().unwrap().to_string());
    }

    // Create relation
    let params = loci::tools::store_relation::StoreRelationParams {
        subject_id: ids[0].clone(),
        predicate: "works_at".into(),
        object_id: ids[1].clone(),
    };

    let result = tools.store_relation(Parameters(params)).await;
    assert!(result.is_ok(), "store_relation failed: {:?}", result.err());

    let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert!(json["id"].is_string());
    assert_eq!(json["deduplicated"], false);
}

// ── end-to-end workflow ─────────────────────────────────────────────────

#[tokio::test]
async fn full_lifecycle_store_recall_inspect_forget() {
    let tools = test_tools();

    // 1. Store
    let store_params = loci::tools::store_memory::StoreMemoryParams {
        content: "The capital of France is Paris".into(),
        r#type: "semantic".into(),
        group: Some("geography".into()),
        scope: Some("global".into()),
        confidence: Some(0.95),
        metadata: None,
        supersedes: None,
    };
    let stored: serde_json::Value =
        serde_json::from_str(&tools.store_memory(Parameters(store_params)).await.unwrap())
            .unwrap();
    let id = stored["id"].as_str().unwrap().to_string();

    // 2. Recall by query
    let recall_params = loci::tools::recall_memory::RecallMemoryParams {
        query: Some("capital France".into()),
        ids: None,
        r#type: Some("semantic".into()),
        scope: None,
        group: Some("geography".into()),
        max_results: Some(1),
        summary_only: None,
        token_budget: None,
        min_confidence: None,
    };
    let recalled: serde_json::Value =
        serde_json::from_str(&tools.recall_memory(Parameters(recall_params)).await.unwrap())
            .unwrap();
    assert!(!recalled["results"].as_array().unwrap().is_empty());

    // 3. Inspect
    let inspect_params = loci::tools::memory_inspect::MemoryInspectParams {
        memory_id: id.clone(),
        include_relations: Some(true),
        include_log: Some(true),
    };
    let inspected: serde_json::Value = serde_json::from_str(
        &tools
            .memory_inspect(Parameters(inspect_params))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(inspected["memory"]["content"].as_str().unwrap(), "The capital of France is Paris");

    // 4. Forget (soft)
    let forget_params = loci::tools::forget_memory::ForgetMemoryParams {
        memory_id: id.clone(),
        reason: Some("outdated".into()),
        hard_delete: None,
    };
    tools.forget_memory(Parameters(forget_params)).await.unwrap();

    // 5. Verify it no longer appears in search
    let recall_params = loci::tools::recall_memory::RecallMemoryParams {
        query: Some("capital France".into()),
        ids: None,
        r#type: None,
        scope: None,
        group: Some("geography".into()),
        max_results: Some(5),
        summary_only: None,
        token_budget: None,
        min_confidence: None,
    };
    let after_forget: serde_json::Value = serde_json::from_str(
        &tools
            .recall_memory(Parameters(recall_params))
            .await
            .unwrap(),
    )
    .unwrap();
    let results = after_forget["results"].as_array().unwrap();
    // Soft-deleted memories should be excluded from search
    for r in results {
        assert_ne!(r["id"].as_str().unwrap(), id);
    }
}
