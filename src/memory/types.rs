//! Core memory type definitions.
//!
//! Defines [`MemoryType`] (the four cognitive memory categories), [`Scope`]
//! (visibility boundaries), [`Memory`] (a full record), and [`EntityRelation`]
//! (graph edges between entity memories).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// The four cognitive memory types, inspired by cognitive science.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Events, decisions, session logs — fast decay, group-scoped by default.
    Episodic,
    /// Facts, knowledge, preferences — slow decay, global-scoped by default.
    Semantic,
    /// Workflows, patterns, how-to guides — slow decay, global-scoped by default.
    Procedural,
    /// People, places, projects, things — slow decay, global-scoped by default.
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
    /// Visible to all groups — used for facts, knowledge, and entities.
    Global,
    /// Visible only within the owning `source_group` — used for episodic events.
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
    /// UUID v7 (time-sortable) primary key.
    pub id: String,
    /// Cognitive category of this memory.
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
    /// The full text content of the memory.
    pub content: String,
    /// Group that owns this memory (e.g. project name). `None` for global memories.
    pub source_group: Option<String>,
    /// Visibility scope — `Global` or `Group`.
    pub scope: Scope,
    /// Confidence score in `[0.0, 1.0]`, decays over time.
    pub confidence: f64,
    /// Number of times this memory has been returned in search results.
    pub access_count: u32,
    /// ISO 8601 timestamp of the last recall, or `None` if never accessed.
    pub last_accessed: Option<String>,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last-modification timestamp.
    pub updated_at: String,
    /// If this memory was replaced, the ID of the replacement (or `"forgotten"`).
    pub superseded_by: Option<String>,
    /// Arbitrary JSON metadata (e.g. `{"summary": true}`).
    pub metadata: Option<serde_json::Value>,
}

/// A directed relationship between two entity memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    /// UUID v7 primary key.
    pub id: String,
    /// ID of the source entity memory.
    pub subject_id: String,
    /// Relationship label (e.g. `"works_at"`, `"manages"`, `"part_of"`).
    pub predicate: String,
    /// ID of the target entity memory.
    pub object_id: String,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
}
