//! MCP `recall_memory` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `recall_memory` MCP tool.
///
/// Provide either `query` (hybrid search) or `ids` (direct hydration), not both.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecallMemoryParams {
    /// Natural language query for hybrid search. Required unless `ids` is provided.
    #[schemars(
        description = "Natural language query to search memories. Required unless 'ids' is provided."
    )]
    pub query: Option<String>,

    /// Specific memory IDs to hydrate (progressive disclosure). Required unless `query` is provided.
    #[schemars(
        description = "Specific memory IDs to hydrate (progressive disclosure). Required unless 'query' is provided."
    )]
    pub ids: Option<Vec<String>>,

    /// Filter by memory type: `"episodic"`, `"semantic"`, `"procedural"`, `"entity"`.
    #[schemars(
        description = "Filter by memory type: 'episodic', 'semantic', 'procedural', 'entity'"
    )]
    pub r#type: Option<String>,

    /// Filter by scope: `"global"` or `"group"`.
    #[schemars(description = "Filter by scope: 'global' or 'group'")]
    pub scope: Option<String>,

    /// Filter by group/project name.
    #[schemars(description = "Filter by group/project name")]
    pub group: Option<String>,

    /// Maximum number of results to return (1–20). Defaults to 5.
    #[schemars(description = "Maximum number of results to return (1-20). Defaults to 5.")]
    pub max_results: Option<usize>,

    /// If `true`, return only compact summaries for token efficiency.
    #[schemars(
        description = "If true, return only summaries (id, type, truncated content, score) for token efficiency. Use recall_memory with ids or memory_inspect to get full details."
    )]
    pub summary_only: Option<bool>,

    /// Token budget limit for the response. Defaults to 4000.
    #[schemars(description = "Token budget limit for the response. Defaults to 4000.")]
    pub token_budget: Option<usize>,

    /// Minimum confidence threshold (0.0–1.0). Defaults to 0.1.
    #[schemars(description = "Minimum confidence threshold (0.0-1.0). Defaults to 0.1.")]
    pub min_confidence: Option<f64>,
}
