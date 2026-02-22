#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// The four cognitive memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    Entity,
}

impl MemoryType {
    /// SQL-compatible string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Entity => "entity",
        }
    }

    /// Default scope for this memory type.
    pub fn default_scope(&self) -> Scope {
        match self {
            Self::Episodic => Scope::Group,
            Self::Semantic | Self::Procedural | Self::Entity => Scope::Global,
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "procedural" => Ok(Self::Procedural),
            "entity" => Ok(Self::Entity),
            _ => Err(format!("unknown memory type: {s}")),
        }
    }
}

/// Visibility scope for a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Global,
    Group,
}

impl Scope {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Group => "group",
        }
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Scope {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "global" => Ok(Self::Global),
            "group" => Ok(Self::Group),
            _ => Err(format!("unknown scope: {s}")),
        }
    }
}

/// A memory record, matching the `memories` table schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
    pub content: String,
    pub source_group: Option<String>,
    pub scope: Scope,
    pub confidence: f64,
    pub access_count: u32,
    pub last_accessed: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub superseded_by: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// An entity relation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    pub id: String,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub created_at: String,
}
