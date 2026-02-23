//! MCP `store_memory` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `store_memory` MCP tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct StoreMemoryParams {
    /// The natural language content of the memory.
    #[schemars(description = "The natural language content of the memory")]
    pub content: String,

    /// Memory type: `"episodic"`, `"semantic"`, `"procedural"`, or `"entity"`.
    #[schemars(
        description = "Memory type: 'episodic' (events/experiences), 'semantic' (facts/knowledge), 'procedural' (how-to/processes), 'entity' (people/places/things)"
    )]
    pub r#type: String,

    /// Optional group/project this memory belongs to.
    #[schemars(description = "Optional group/project this memory belongs to")]
    pub group: Option<String>,

    /// Visibility scope: `"global"` or `"group"`. Defaults based on type.
    #[schemars(
        description = "Visibility scope: 'global' (all groups) or 'group' (only this group). Defaults based on type."
    )]
    pub scope: Option<String>,

    /// Initial confidence score in `[0.0, 1.0]`. Defaults to `1.0`.
    #[schemars(description = "Initial confidence score 0.0-1.0. Defaults to 1.0.")]
    pub confidence: Option<f64>,

    /// Optional JSON metadata blob for type-specific data.
    #[schemars(description = "Optional JSON metadata blob for type-specific data")]
    pub metadata: Option<serde_json::Value>,

    /// ID of memory this replaces; the old memory will be marked superseded.
    #[schemars(
        description = "ID of memory this replaces. The old memory's superseded_by will be set to the new ID."
    )]
    pub supersedes: Option<String>,
}
