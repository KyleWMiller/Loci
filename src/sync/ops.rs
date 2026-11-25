//! Wire format of replicated operations (`v: 1`).

use serde::{Deserialize, Serialize};

/// Current op format version.
pub const OP_VERSION: u32 = 1;

/// The replicated content of a memory (no access stats, no `superseded_by`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PutMemory {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    pub content: String,
    pub source_group: Option<String>,
    pub scope: String,
    pub confidence: f64,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// A replicated relation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PutRelation {
    pub id: String,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub created_at: String,
}

/// The operation payload, tagged by `op`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Op {
    #[serde(rename = "memory.put")]
    MemoryPut { memory: PutMemory },
    #[serde(rename = "memory.supersede")]
    MemorySupersede { old_id: String, new_id: String },
    /// Carries only the ID — never content or a reason.
    #[serde(rename = "memory.forget")]
    MemoryForget { id: String, hard: bool },
    #[serde(rename = "relation.put")]
    RelationPut { relation: PutRelation },
    #[serde(rename = "relation.delete")]
    RelationDelete { id: String },
    /// A scrubbed op: a `memory.put` whose memory was hard-deleted. Ignored on apply.
    #[serde(rename = "noop")]
    Noop,
}

impl Op {
    /// Short kind string, equal to the serialized `op` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Op::MemoryPut { .. } => "memory.put",
            Op::MemorySupersede { .. } => "memory.supersede",
            Op::MemoryForget { .. } => "memory.forget",
            Op::RelationPut { .. } => "relation.put",
            Op::RelationDelete { .. } => "relation.delete",
            Op::Noop => "noop",
        }
    }

    /// The memory this op is about, if any (used to scrub on hard delete).
    pub fn memory_id(&self) -> Option<&str> {
        match self {
            Op::MemoryPut { memory } => Some(&memory.id),
            Op::MemoryForget { id, .. } => Some(id),
            _ => None,
        }
    }
}

/// One line of a sync file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpEnvelope {
    pub v: u32,
    /// UUID v7 identifying the op.
    pub op_id: String,
    /// RFC 3339 UTC timestamp.
    pub ts: String,
    /// `machine_id` of the machine that created the op.
    pub origin: String,
    #[serde(flatten)]
    pub op: Op,
}
