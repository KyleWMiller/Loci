use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct StoreRelationParams {
    #[schemars(description = "ID of the subject entity memory")]
    pub subject_id: String,

    #[schemars(description = "Relationship predicate (e.g. 'works_at', 'manages', 'part_of')")]
    pub predicate: String,

    #[schemars(description = "ID of the object entity memory")]
    pub object_id: String,
}
