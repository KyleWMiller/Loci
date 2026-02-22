use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemoryStatsParams {
    #[schemars(description = "Optional group to filter stats by")]
    pub group: Option<String>,
}
