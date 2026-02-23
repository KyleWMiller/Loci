//! MCP `memory_stats` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `memory_stats` MCP tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryStatsParams {
    /// Optional group name to filter statistics by.
    #[schemars(description = "Optional group to filter stats by")]
    pub group: Option<String>,
}
