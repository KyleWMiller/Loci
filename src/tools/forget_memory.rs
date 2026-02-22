use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ForgetMemoryParams {
    #[schemars(description = "ID of the memory to delete")]
    pub id: Option<String>,

    #[schemars(description = "Delete all memories matching this type")]
    pub r#type: Option<String>,

    #[schemars(description = "Delete all memories in this group")]
    pub group: Option<String>,

    #[schemars(description = "Must be true to confirm deletion. Safety gate.")]
    pub confirm: Option<bool>,
}
