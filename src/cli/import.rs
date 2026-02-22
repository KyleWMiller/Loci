use anyhow::{Context, Result};
use rusqlite::params;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

use crate::config::LociConfig;
use crate::memory::types::{EntityRelation, Memory};

/// Import format — matches export output.
#[derive(Debug, Deserialize)]
struct ImportData {
    memories: Vec<Memory>,
    #[serde(default)]
    relations: Vec<EntityRelation>,
}

/// Import memories from a JSON file.
///
/// Re-embeds each memory using the local ONNX model. Skips memories whose ID
/// already exists in the database. Relations are re-created if both endpoints exist.
pub async fn import(config: &LociConfig, file: &Path) -> Result<()> {
    let json = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read import file: {}", file.display()))?;

    let data: ImportData =
        serde_json::from_str(&json).context("failed to parse import JSON")?;

    let db_path = config.resolved_db_path();
    let mut conn = crate::db::open_database(&db_path)?;

    // Create embedding provider
    let provider = crate::embedding::create_provider(&config.embedding)?;
    let embedding_provider: Arc<dyn crate::embedding::EmbeddingProvider> = Arc::from(provider);

    let mut imported = 0u64;
    let mut skipped = 0u64;

    println!(
        "Importing {} memories and {} relations...",
        data.memories.len(),
        data.relations.len()
    );

    for memory in &data.memories {
        // Check if ID already exists
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
            params![memory.id],
            |row| row.get(0),
        )?;

        if exists {
            skipped += 1;
            continue;
        }

        // Re-embed the content
        let ep = Arc::clone(&embedding_provider);
        let content = memory.content.clone();
        let embedding = tokio::task::spawn_blocking(move || ep.embed(&content)).await??;

        // Store using the full write path
        crate::memory::store::store_memory(
            &mut conn,
            &memory.content,
            memory.memory_type,
            memory.scope,
            memory.source_group.as_deref(),
            memory.confidence,
            memory.metadata.as_ref(),
            None, // don't re-apply supersession chains
            &embedding,
            // Use a threshold of 1.0 to effectively disable dedup during import
            1.0,
        )?;

        imported += 1;
    }

    // Re-create relations where both endpoints exist
    let mut relations_created = 0u64;
    let mut relations_skipped = 0u64;

    for rel in &data.relations {
        // Check both endpoints exist
        let subject_exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
            params![rel.subject_id],
            |row| row.get(0),
        )?;
        let object_exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
            params![rel.object_id],
            |row| row.get(0),
        )?;

        if subject_exists && object_exists {
            match crate::memory::relations::store_relation(
                &conn,
                &rel.subject_id,
                &rel.predicate,
                &rel.object_id,
            ) {
                Ok(_) => relations_created += 1,
                Err(e) => {
                    eprintln!("Warning: failed to create relation: {e}");
                    relations_skipped += 1;
                }
            }
        } else {
            relations_skipped += 1;
        }
    }

    println!("Import complete:");
    println!("  Memories imported: {imported}");
    println!("  Memories skipped:  {skipped} (already exist)");
    println!("  Relations created: {relations_created}");
    if relations_skipped > 0 {
        println!("  Relations skipped: {relations_skipped}");
    }

    Ok(())
}
