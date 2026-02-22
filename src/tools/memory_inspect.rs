use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryInspectParams {
    #[schemars(description = "ID of the memory to inspect")]
    pub id: String,

    #[schemars(description = "If true, include related entities in the response")]
    pub include_relations: Option<bool>,
}
