pub mod forget_memory;
pub mod memory_inspect;
pub mod memory_stats;
pub mod recall_memory;
pub mod store_memory;
pub mod store_relation;

use forget_memory::ForgetMemoryParams;
use memory_inspect::MemoryInspectParams;
use memory_stats::MemoryStatsParams;
use recall_memory::RecallMemoryParams;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use store_memory::StoreMemoryParams;
use store_relation::StoreRelationParams;

use crate::config::LociConfig;
use crate::embedding::EmbeddingProvider;
use crate::memory::types::{MemoryType, Scope};

/// The Loci MCP tool handler. Holds shared state (db connection, embedding provider,
/// config) and exposes all MCP tools via the `#[tool_router]` macro.
#[derive(Clone)]
pub struct LociTools {
    tool_router: ToolRouter<Self>,
    db: Arc<Mutex<Connection>>,
    embedding: Arc<dyn EmbeddingProvider>,
    config: Arc<LociConfig>,
}

#[tool_router]
impl LociTools {
    pub fn new(
        db: Arc<Mutex<Connection>>,
        embedding: Arc<dyn EmbeddingProvider>,
        config: Arc<LociConfig>,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            db,
            embedding,
            config,
        }
    }

    /// Store a new memory in the cognitive memory system.
    #[tool(description = "Store a new memory. Types: episodic (events/experiences), semantic (facts/knowledge), procedural (how-to/processes), entity (people/places/things).")]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<String, String> {
        // 1. Validate inputs
        let memory_type: MemoryType = params.r#type.parse().map_err(|e: String| e)?;

        let scope = match &params.scope {
            Some(s) => s.parse::<Scope>().map_err(|e: String| e)?,
            None => memory_type.default_scope(),
        };

        let confidence = params.confidence.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&confidence) {
            return Err("confidence must be between 0.0 and 1.0".into());
        }

        if params.content.is_empty() {
            return Err("content must not be empty".into());
        }

        let group = params
            .group
            .as_deref()
            .unwrap_or(&self.config.storage.default_group);

        tracing::info!(
            content_len = params.content.len(),
            memory_type = %memory_type,
            scope = %scope,
            group = %group,
            "store_memory called"
        );

        // 2. Embed content (CPU-heavy → spawn_blocking)
        let embedding_provider = Arc::clone(&self.embedding);
        let content_for_embed = params.content.clone();
        let embedding = tokio::task::spawn_blocking(move || {
            embedding_provider.embed(&content_for_embed)
        })
        .await
        .map_err(|e| format!("embedding task failed: {e}"))?
        .map_err(|e| format!("embedding failed: {e}"))?;

        // 3. Run write path (sync DB ops → spawn_blocking)
        let db = Arc::clone(&self.db);
        let dedup_threshold = self.config.retrieval.dedup_threshold;
        let content = params.content;
        let metadata = params.metadata;
        let supersedes = params.supersedes;
        let group_owned = group.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let mut conn = db
                .lock()
                .map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::store::store_memory(
                &mut conn,
                &content,
                memory_type,
                scope,
                Some(&group_owned),
                confidence,
                metadata.as_ref(),
                supersedes.as_deref(),
                &embedding,
                dedup_threshold,
            )
        })
        .await
        .map_err(|e| format!("db task failed: {e}"))?
        .map_err(|e| format!("store failed: {e}"))?;

        tracing::info!(
            id = %result.id,
            deduplicated = result.deduplicated,
            "memory stored"
        );

        serde_json::to_string(&result).map_err(|e| format!("serialization failed: {e}"))
    }

    /// Search and retrieve memories using natural language queries.
    #[tool(description = "Search memories by natural language query. Returns ranked results using hybrid vector + keyword search. Provide 'query' for search or 'ids' for direct hydration.")]
    async fn recall_memory(
        &self,
        Parameters(params): Parameters<RecallMemoryParams>,
    ) -> Result<String, String> {
        // Validate: at least one of query or ids must be provided
        if params.query.is_none() && params.ids.is_none() {
            return Err("either 'query' or 'ids' must be provided".into());
        }

        let group = params
            .group
            .as_deref()
            .unwrap_or(&self.config.storage.default_group)
            .to_string();
        let summary_only = params.summary_only.unwrap_or(false);

        // ID hydration mode
        if let Some(ids) = params.ids {
            tracing::info!(count = ids.len(), "recall_memory: hydrating by IDs");
            let db = Arc::clone(&self.db);
            let response = tokio::task::spawn_blocking(move || {
                let conn = db.lock().map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
                crate::memory::search::recall_by_ids(&conn, &ids)
            })
            .await
            .map_err(|e| format!("task failed: {e}"))?
            .map_err(|e| format!("recall failed: {e}"))?;

            if summary_only {
                let summary = crate::memory::search::to_summary(&response);
                return serde_json::to_string(&summary)
                    .map_err(|e| format!("serialization failed: {e}"));
            }
            return serde_json::to_string(&response)
                .map_err(|e| format!("serialization failed: {e}"));
        }

        // Query search mode
        let query = params.query.unwrap(); // safe: validated above
        tracing::info!(query = %query, "recall_memory: hybrid search");

        // Embed the query
        let embedding_provider = Arc::clone(&self.embedding);
        let query_for_embed = query.clone();
        let query_embedding = tokio::task::spawn_blocking(move || {
            embedding_provider.embed(&query_for_embed)
        })
        .await
        .map_err(|e| format!("embedding task failed: {e}"))?
        .map_err(|e| format!("embedding failed: {e}"))?;

        // Parse optional filters
        let memory_type = params
            .r#type
            .as_deref()
            .map(|t| t.parse::<MemoryType>())
            .transpose()
            .map_err(|e| e)?;

        let scope = params
            .scope
            .as_deref()
            .map(|s| s.parse::<Scope>())
            .transpose()
            .map_err(|e| e)?;

        let max_results = params
            .max_results
            .unwrap_or(self.config.retrieval.default_max_results)
            .clamp(1, 20);

        let token_budget = params
            .token_budget
            .unwrap_or(self.config.retrieval.recall_token_budget);

        let min_confidence = params.min_confidence.unwrap_or(0.1);

        let rrf_k = self.config.retrieval.rrf_k;

        let filter = crate::memory::search::SearchFilter {
            memory_type,
            scope,
            group,
            min_confidence,
        };

        let search_config = crate::memory::search::SearchConfig {
            max_results,
            token_budget,
            rrf_k,
        };

        // Run hybrid search
        let db = Arc::clone(&self.db);
        let response = tokio::task::spawn_blocking(move || {
            let conn = db.lock().map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::search::recall_by_query(
                &conn,
                &query_embedding,
                &query,
                &filter,
                &search_config,
            )
        })
        .await
        .map_err(|e| format!("search task failed: {e}"))?
        .map_err(|e| format!("search failed: {e}"))?;

        tracing::info!(
            results = response.results.len(),
            total_matched = response.total_matched,
            token_estimate = response.token_estimate,
            "recall_memory complete"
        );

        if summary_only {
            let summary = crate::memory::search::to_summary(&response);
            return serde_json::to_string(&summary)
                .map_err(|e| format!("serialization failed: {e}"));
        }

        serde_json::to_string(&response).map_err(|e| format!("serialization failed: {e}"))
    }

    /// Forget a memory by ID (soft-supersede or hard delete).
    #[tool(description = "Forget a memory by ID. Soft delete (default) marks it as superseded. Hard delete permanently removes it from all tables including vectors and FTS index.")]
    async fn forget_memory(
        &self,
        Parameters(params): Parameters<ForgetMemoryParams>,
    ) -> Result<String, String> {
        if params.memory_id.is_empty() {
            return Err("memory_id must not be empty".into());
        }

        let hard_delete = params.hard_delete.unwrap_or(false);
        tracing::info!(
            id = %params.memory_id,
            hard_delete = hard_delete,
            "forget_memory called"
        );

        let db = Arc::clone(&self.db);
        let memory_id = params.memory_id;
        let reason = params.reason;

        let result = tokio::task::spawn_blocking(move || {
            let mut conn = db
                .lock()
                .map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::forget::forget_memory(
                &mut conn,
                &memory_id,
                reason.as_deref(),
                hard_delete,
            )
        })
        .await
        .map_err(|e| format!("task failed: {e}"))?
        .map_err(|e| format!("forget failed: {e}"))?;

        tracing::info!(
            id = %result.id,
            hard_deleted = result.hard_deleted,
            "memory forgotten"
        );

        serde_json::to_string(&result).map_err(|e| format!("serialization failed: {e}"))
    }

    /// Get statistics about the memory store.
    #[tool(description = "Get memory store statistics: counts by type and scope, entity relations count, storage size, oldest/newest timestamps.")]
    async fn memory_stats(
        &self,
        Parameters(params): Parameters<MemoryStatsParams>,
    ) -> Result<String, String> {
        tracing::info!("memory_stats called");

        let db = Arc::clone(&self.db);
        let group = params.group;
        let db_path = self.config.resolved_db_path();

        let result = tokio::task::spawn_blocking(move || {
            let conn = db
                .lock()
                .map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::stats::memory_stats(&conn, group.as_deref(), Some(&db_path))
        })
        .await
        .map_err(|e| format!("task failed: {e}"))?
        .map_err(|e| format!("stats failed: {e}"))?;

        serde_json::to_string(&result).map_err(|e| format!("serialization failed: {e}"))
    }

    /// Inspect a specific memory by ID.
    #[tool(description = "Inspect a memory by ID. Returns full content, metadata, confidence, access history, and optionally related entities and audit log.")]
    async fn memory_inspect(
        &self,
        Parameters(params): Parameters<MemoryInspectParams>,
    ) -> Result<String, String> {
        tracing::info!(id = %params.memory_id, "memory_inspect called");

        let include_relations = params.include_relations.unwrap_or(true);
        let include_log = params.include_log.unwrap_or(false);
        let memory_id = params.memory_id;

        let db = Arc::clone(&self.db);
        let response = tokio::task::spawn_blocking(move || {
            let conn = db.lock().map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::search::inspect_memory(&conn, &memory_id, include_relations, include_log)
        })
        .await
        .map_err(|e| format!("task failed: {e}"))?
        .map_err(|e| format!("inspect failed: {e}"))?;

        serde_json::to_string(&response).map_err(|e| format!("serialization failed: {e}"))
    }

    /// Store a relationship between two entity memories.
    #[tool(description = "Create a relationship between two entity memories (e.g. 'works_at', 'manages', 'part_of'). Both IDs must refer to entity-type memories. Idempotent on (subject, predicate, object).")]
    async fn store_relation(
        &self,
        Parameters(params): Parameters<StoreRelationParams>,
    ) -> Result<String, String> {
        if params.subject_id.is_empty() {
            return Err("subject_id must not be empty".into());
        }
        if params.predicate.is_empty() {
            return Err("predicate must not be empty".into());
        }
        if params.object_id.is_empty() {
            return Err("object_id must not be empty".into());
        }

        tracing::info!(
            subject = %params.subject_id,
            predicate = %params.predicate,
            object = %params.object_id,
            "store_relation called"
        );

        let db = Arc::clone(&self.db);
        let subject_id = params.subject_id;
        let predicate = params.predicate;
        let object_id = params.object_id;

        let result = tokio::task::spawn_blocking(move || {
            let conn = db
                .lock()
                .map_err(|e| anyhow::anyhow!("db lock poisoned: {e}"))?;
            crate::memory::relations::store_relation(&conn, &subject_id, &predicate, &object_id)
        })
        .await
        .map_err(|e| format!("task failed: {e}"))?
        .map_err(|e| format!("store_relation failed: {e}"))?;

        tracing::info!(
            id = %result.id,
            deduplicated = result.deduplicated,
            "relation stored"
        );

        serde_json::to_string(&result).map_err(|e| format!("serialization failed: {e}"))
    }
}

#[tool_handler]
impl ServerHandler for LociTools {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo {
            instructions: Some(
                "Loci is a cognitive memory server. Use store_memory to save memories, \
                 recall_memory to search, and memory_inspect to view details."
                    .into(),
            ),
            capabilities: rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
