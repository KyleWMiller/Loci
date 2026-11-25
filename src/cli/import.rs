//! CLI `import` command — restore memories from a JSON export file.
//!
//! Import is ID-preserving and idempotent: memory IDs, `superseded_by` chains and
//! relation IDs are kept exactly as exported, memories that already exist are skipped
//! without re-embedding, and running the same import twice changes nothing.

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::cli::export::ExportData;
use crate::config::LociConfig;
use crate::memory::relations::{RelationInsert, store_relation_with_id};
use crate::memory::store::{NewMemory, store_new_memory};

/// Counts reported by [`import_data`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ImportSummary {
    /// Memories inserted under their original ID.
    pub imported: u64,
    /// Memories skipped because the ID already exists (not re-embedded).
    pub skipped: u64,
    /// Memories skipped because the ID was hard-deleted locally (tombstone).
    pub tombstoned: u64,
    /// `superseded_by` links restored on newly imported memories.
    pub supersessions_applied: u64,
    /// Relations inserted under their original ID.
    pub relations_created: u64,
    /// Relations that already existed (same ID or same triple).
    pub relations_existing: u64,
    /// Relations skipped because an endpoint is missing or invalid.
    pub relations_skipped: u64,
}

/// Apply an export to `conn`. `embed` produces the embedding for a memory's content
/// and is only called for memories that are actually inserted.
pub fn import_data(
    conn: &mut Connection,
    data: &ExportData,
    embed: &mut dyn FnMut(&str) -> Result<Vec<f32>>,
) -> Result<ImportSummary> {
    let mut summary = ImportSummary::default();
    // Chains are applied after every put so a target may appear later in the file.
    let mut pending_supersession: HashMap<&str, &str> = HashMap::new();

    for memory in &data.memories {
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
            params![memory.id],
            |row| row.get(0),
        )?;
        if exists {
            summary.skipped += 1;
            continue;
        }

        let hard_deleted: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM tombstones WHERE memory_id = ?1 AND hard = 1",
            params![memory.id],
            |row| row.get(0),
        )?;
        if hard_deleted {
            summary.tombstoned += 1;
            continue;
        }

        let embedding = embed(&memory.content)?;

        let mut new = NewMemory::new(
            &memory.content,
            memory.memory_type,
            memory.scope,
            memory.source_group.as_deref(),
            memory.confidence,
            memory.metadata.as_ref(),
        );
        new.id = Some(&memory.id);
        new.created_at = Some(&memory.created_at);
        new.updated_at = Some(&memory.updated_at);
        // Dedup off: import preserves identity, it never merges.
        store_new_memory(conn, &new, &embedding, None)?;
        summary.imported += 1;

        if let Some(target) = memory.superseded_by.as_deref() {
            pending_supersession.insert(&memory.id, target);
        }
    }

    // Restore supersession (including soft-forgotten = "forgotten") for imported rows.
    for (id, target) in &pending_supersession {
        conn.execute(
            "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
            params![target, id],
        )?;
        summary.supersessions_applied += 1;
    }

    for rel in &data.relations {
        match store_relation_with_id(
            conn,
            &rel.id,
            &rel.subject_id,
            &rel.predicate,
            &rel.object_id,
            &rel.created_at,
        ) {
            Ok(RelationInsert::Inserted) => summary.relations_created += 1,
            Ok(RelationInsert::Existing) => summary.relations_existing += 1,
            Err(e) => {
                tracing::warn!(relation = %rel.id, error = %e, "skipping relation on import");
                summary.relations_skipped += 1;
            }
        }
    }

    Ok(summary)
}

/// Import memories from a JSON file.
///
/// Re-embeds each newly inserted memory using the local ONNX model. Existing IDs are
/// skipped without re-embedding. Relations keep their IDs and are created when both
/// endpoints exist.
pub async fn import(config: &LociConfig, file: &Path) -> Result<()> {
    let json = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read import file: {}", file.display()))?;

    let data: ExportData = serde_json::from_str(&json).context("failed to parse import JSON")?;

    let db_path = config.resolved_db_path();
    let mut conn = crate::db::open_database(&db_path)?;

    let provider = crate::embedding::create_provider(&config.embedding)?;
    let embedding_provider: Arc<dyn crate::embedding::EmbeddingProvider> = Arc::from(provider);

    println!(
        "Importing {} memories and {} relations...",
        data.memories.len(),
        data.relations.len()
    );

    let summary = tokio::task::spawn_blocking(move || {
        import_data(&mut conn, &data, &mut |text| embedding_provider.embed(text))
    })
    .await??;

    println!("Import complete:");
    println!("  Memories imported: {}", summary.imported);
    println!("  Memories skipped:  {} (already exist)", summary.skipped);
    if summary.tombstoned > 0 {
        println!(
            "  Memories skipped:  {} (hard-deleted locally)",
            summary.tombstoned
        );
    }
    println!(
        "  Supersessions restored: {}",
        summary.supersessions_applied
    );
    println!("  Relations created: {}", summary.relations_created);
    if summary.relations_existing > 0 {
        println!(
            "  Relations already present: {}",
            summary.relations_existing
        );
    }
    if summary.relations_skipped > 0 {
        println!("  Relations skipped: {}", summary.relations_skipped);
    }

    Ok(())
}
