//! MCP `store_relation` tool parameter definition.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `store_relation` MCP tool.
///
/// Creates a directed relationship between two entity-type memories.
/// Idempotent on the (subject, predicate, object) triple.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct StoreRelationParams {
    /// ID of the source entity memory.
    #[schemars(description = "ID of the subject entity memory")]
    pub subject_id: String,

    /// Relationship label (e.g. `"works_at"`, `"manages"`, `"part_of"`).
    #[schemars(description = "Relationship predicate (e.g. 'works_at', 'manages', 'part_of')")]
    pub predicate: String,

    /// ID of the target entity memory.
    #[schemars(description = "ID of the object entity memory")]
    pub object_id: String,
}
