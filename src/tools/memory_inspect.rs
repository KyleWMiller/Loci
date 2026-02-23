//! MCP `memory_inspect` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `memory_inspect` MCP tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryInspectParams {
    /// ID of the memory to inspect.
    #[schemars(description = "ID of the memory to inspect")]
    pub memory_id: String,

    /// Include outbound entity relations (default: `true`).
    #[schemars(description = "If true, include related entities in the response. Defaults to true.")]
    pub include_relations: Option<bool>,

    /// Include audit log entries for this memory (default: `false`).
    #[schemars(description = "If true, include audit log entries for this memory. Defaults to false.")]
    pub include_log: Option<bool>,
}
