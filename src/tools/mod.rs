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
    #[tool(description = "Search memories by natural language query. Returns ranked results using hybrid vector + keyword search.")]
    async fn recall_memory(
        &self,
        Parameters(params): Parameters<RecallMemoryParams>,
    ) -> Result<String, String> {
        tracing::info!(query = %params.query, "recall_memory called (stub)");
        Ok(serde_json::json!({
            "memories": [],
            "total": 0,
            "message": "recall_memory is a stub — implementation coming in M3"
        })
        .to_string())
    }

    /// Delete one or more memories.
    #[tool(description = "Delete memories by ID, type, or group. Requires confirm=true as a safety gate.")]
    async fn forget_memory(
        &self,
        Parameters(_params): Parameters<ForgetMemoryParams>,
    ) -> Result<String, String> {
        tracing::info!("forget_memory called (stub)");
        Ok(serde_json::json!({
            "status": "not_implemented",
            "message": "forget_memory is a stub — implementation coming in M5"
        })
        .to_string())
    }

    /// Get statistics about the memory store.
    #[tool(description = "Get memory store statistics: counts by type, groups, storage size.")]
    async fn memory_stats(
        &self,
        Parameters(_params): Parameters<MemoryStatsParams>,
    ) -> Result<String, String> {
        tracing::info!("memory_stats called (stub)");
        Ok(serde_json::json!({
            "total_memories": 0,
            "by_type": {
                "episodic": 0,
                "semantic": 0,
                "procedural": 0,
                "entity": 0
            },
            "groups": [],
            "message": "memory_stats is a stub — implementation coming in M5"
        })
        .to_string())
    }

    /// Inspect a specific memory by ID.
    #[tool(description = "Inspect a memory by ID. Returns full content, metadata, and optionally related entities.")]
    async fn memory_inspect(
        &self,
        Parameters(params): Parameters<MemoryInspectParams>,
    ) -> Result<String, String> {
        tracing::info!(id = %params.id, "memory_inspect called (stub)");
        Ok(serde_json::json!({
            "status": "not_implemented",
            "message": "memory_inspect is a stub — implementation coming in M5"
        })
        .to_string())
    }

    /// Store a relationship between two entity memories.
    #[tool(description = "Create a relationship between two entity memories (e.g. 'works_at', 'manages', 'part_of').")]
    async fn store_relation(
        &self,
        Parameters(params): Parameters<StoreRelationParams>,
    ) -> Result<String, String> {
        tracing::info!(
            subject = %params.subject_id,
            predicate = %params.predicate,
            object = %params.object_id,
            "store_relation called (stub)"
        );
        Ok(serde_json::json!({
            "status": "not_implemented",
            "message": "store_relation is a stub — implementation coming in M4"
        })
        .to_string())
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
