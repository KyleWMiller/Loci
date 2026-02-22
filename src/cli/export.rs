use anyhow::Result;
use rusqlite::params;
use serde::Serialize;

use crate::config::LociConfig;
use crate::memory::types::{EntityRelation, Memory};

/// Export format — wraps all memories and relations.
#[derive(Debug, Serialize)]
struct ExportData {
    memories: Vec<Memory>,
    relations: Vec<EntityRelation>,
}

/// Export all memories and relations as JSON to stdout.
pub fn export(config: &LociConfig) -> Result<()> {
    let db_path = config.resolved_db_path();
    let conn = crate::db::open_database(&db_path)?;

    // Fetch all memories
    let mut stmt = conn.prepare(
        "SELECT id, type, content, source_group, scope, confidence, access_count, \
         last_accessed, created_at, updated_at, superseded_by, metadata \
         FROM memories ORDER BY created_at",
    )?;

    let memories: Vec<Memory> = stmt
        .query_map([], |row| {
            let metadata_str: Option<String> = row.get(11)?;
            let memory_type_str: String = row.get(1)?;
            let scope_str: String = row.get(4)?;
            Ok(Memory {
                id: row.get(0)?,
                memory_type: memory_type_str
                    .parse()
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                content: row.get(2)?,
                source_group: row.get(3)?,
                scope: scope_str
                    .parse()
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                confidence: row.get(5)?,
                access_count: row.get(6)?,
                last_accessed: row.get(7)?,
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
                superseded_by: row.get(10)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Fetch all relations
    let mut stmt = conn.prepare(
        "SELECT id, subject_id, predicate, object_id, created_at \
         FROM entity_relations ORDER BY created_at",
    )?;

    let relations: Vec<EntityRelation> = stmt
        .query_map(params![], |row| {
            Ok(EntityRelation {
                id: row.get(0)?,
                subject_id: row.get(1)?,
                predicate: row.get(2)?,
                object_id: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let data = ExportData {
        memories,
        relations,
    };

    let json = serde_json::to_string_pretty(&data)?;
    println!("{json}");

    eprintln!(
        "Exported {} memories and {} relations.",
        data.memories.len(),
        data.relations.len()
    );

    Ok(())
}
