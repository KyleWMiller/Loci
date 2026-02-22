use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryInspectParams {
    #[schemars(description = "ID of the memory to inspect")]
    pub memory_id: String,

    #[schemars(description = "If true, include related entities in the response. Defaults to true.")]
    pub include_relations: Option<bool>,

    #[schemars(description = "If true, include audit log entries for this memory. Defaults to false.")]
    pub include_log: Option<bool>,
}
