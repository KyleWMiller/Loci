use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct StoreMemoryParams {
    #[schemars(description = "The natural language content of the memory")]
    pub content: String,

    #[schemars(
        description = "Memory type: 'episodic' (events/experiences), 'semantic' (facts/knowledge), 'procedural' (how-to/processes), 'entity' (people/places/things)"
    )]
    pub r#type: String,

    #[schemars(description = "Optional group/project this memory belongs to")]
    pub group: Option<String>,

    #[schemars(
        description = "Visibility scope: 'global' (all groups) or 'group' (only this group). Defaults based on type."
    )]
    pub scope: Option<String>,

    #[schemars(description = "Initial confidence score 0.0-1.0. Defaults to 1.0.")]
    pub confidence: Option<f64>,

    #[schemars(description = "Optional JSON metadata blob for type-specific data")]
    pub metadata: Option<serde_json::Value>,

    #[schemars(
        description = "ID of memory this replaces. The old memory's superseded_by will be set to the new ID."
    )]
    pub supersedes: Option<String>,
}
