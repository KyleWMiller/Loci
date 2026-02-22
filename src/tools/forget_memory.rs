use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ForgetMemoryParams {
    #[schemars(description = "ID of the memory to forget")]
    pub memory_id: String,

    #[schemars(description = "Why this memory is being forgotten")]
    pub reason: Option<String>,

    #[schemars(description = "Permanently delete instead of soft-supersede (default: false)")]
    pub hard_delete: Option<bool>,
}
