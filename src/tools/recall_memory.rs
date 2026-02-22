use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecallMemoryParams {
    #[schemars(description = "Natural language query to search memories")]
    pub query: String,

    #[schemars(description = "Filter by memory type: 'episodic', 'semantic', 'procedural', 'entity'")]
    pub r#type: Option<String>,

    #[schemars(description = "Filter by group/project name")]
    pub group: Option<String>,

    #[schemars(description = "Maximum number of results to return. Defaults to 5.")]
    pub max_results: Option<usize>,

    #[schemars(
        description = "If true, return only summaries (id, type, truncated content) for token efficiency. Use memory_inspect to get full details."
    )]
    pub summary_only: Option<bool>,

    #[schemars(description = "Token budget limit for the response. Defaults to 4000.")]
    pub token_budget: Option<usize>,
}
