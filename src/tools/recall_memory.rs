use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecallMemoryParams {
    #[schemars(
        description = "Natural language query to search memories. Required unless 'ids' is provided."
    )]
    pub query: Option<String>,

    #[schemars(
        description = "Specific memory IDs to hydrate (progressive disclosure). Required unless 'query' is provided."
    )]
    pub ids: Option<Vec<String>>,

    #[schemars(
        description = "Filter by memory type: 'episodic', 'semantic', 'procedural', 'entity'"
    )]
    pub r#type: Option<String>,

    #[schemars(description = "Filter by scope: 'global' or 'group'")]
    pub scope: Option<String>,

    #[schemars(description = "Filter by group/project name")]
    pub group: Option<String>,

    #[schemars(description = "Maximum number of results to return (1-20). Defaults to 5.")]
    pub max_results: Option<usize>,

    #[schemars(
        description = "If true, return only summaries (id, type, truncated content, score) for token efficiency. Use recall_memory with ids or memory_inspect to get full details."
    )]
    pub summary_only: Option<bool>,

    #[schemars(description = "Token budget limit for the response. Defaults to 4000.")]
    pub token_budget: Option<usize>,

    #[schemars(description = "Minimum confidence threshold (0.0-1.0). Defaults to 0.1.")]
    pub min_confidence: Option<f64>,
}
