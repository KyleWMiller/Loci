//! MCP `forget_memory` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `forget_memory` MCP tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ForgetMemoryParams {
    /// ID of the memory to forget.
    #[schemars(description = "ID of the memory to forget")]
    pub memory_id: String,

    /// Optional reason for forgetting (recorded in audit log).
    #[schemars(description = "Why this memory is being forgotten")]
    pub reason: Option<String>,

    /// Permanently delete instead of soft-supersede (default: `false`).
    #[schemars(description = "Permanently delete instead of soft-supersede (default: false)")]
    pub hard_delete: Option<bool>,
}
