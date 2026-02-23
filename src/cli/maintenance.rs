//! CLI maintenance commands — `compact` and `cleanup` for memory lifecycle management.

use anyhow::Result;

use crate::config::LociConfig;
use crate::memory::maintenance;

/// Run full compaction cycle: decay + compact + promote.
///
/// Async because compaction and promotion need the embedding provider.
pub async fn compact(config: &LociConfig) -> Result<()> {
    let db_path = config.resolved_db_path();
    let mut conn = crate::db::open_database(&db_path)?;
    let embedding = crate::embedding::create_provider(&config.embedding)?;

    // 1. Confidence decay
    println!("Applying confidence decay...");
    let decay_result = maintenance::apply_decay(&conn, &config.maintenance)?;

    let total_decayed: usize = decay_result.affected_by_type.values().sum();
    if total_decayed > 0 {
        println!("  Decayed {total_decayed} memories:");
        for (mem_type, count) in &decay_result.affected_by_type {
            if *count > 0 {
                println!("    {mem_type}: {count}");
            }
        }
    } else {
        println!("  No memories to decay.");
    }

    // 2. Episodic compaction
    println!("Running episodic compaction...");
    let compact_result =
        maintenance::compact_episodic(&mut conn, embedding.as_ref(), &config.maintenance)?;

    if compact_result.summaries_created > 0 {
        println!(
            "  Compacted {} memories across {} groups into {} summaries.",
            compact_result.memories_compacted,
            compact_result.groups_compacted,
            compact_result.summaries_created,
        );
    } else {
        println!("  No episodic groups eligible for compaction.");
    }

    // 3. Episodic-to-semantic promotion
    println!("Checking for episodic-to-semantic promotions...");
    let promote_result = maintenance::promote_episodic_to_semantic(
        &mut conn,
        embedding.as_ref(),
        &config.maintenance,
    )?;

    if promote_result.semantics_created > 0 {
        println!(
            "  Found {} clusters, created {} semantic memories.",
            promote_result.clusters_found, promote_result.semantics_created,
        );
    } else {
        println!("  No episodic clusters eligible for promotion.");
    }

    println!("Compaction complete.");
    Ok(())
}

/// Run cleanup of stale, low-confidence memories.
pub fn cleanup(config: &LociConfig, dry_run: bool) -> Result<()> {
    let db_path = config.resolved_db_path();
    let mut conn = crate::db::open_database(&db_path)?;

    let result = maintenance::cleanup_stale(&mut conn, &config.maintenance, dry_run)?;

    if result.candidates.is_empty() {
        println!("No stale memories found.");
        return Ok(());
    }

    if dry_run {
        println!(
            "Found {} candidate(s) for cleanup (dry run — nothing deleted):\n",
            result.candidates.len()
        );
        println!(
            "{:<38} {:<12} {:<10} {}",
            "ID", "Type", "Confidence", "Preview"
        );
        println!("{}", "-".repeat(90));
        for c in &result.candidates {
            println!(
                "{:<38} {:<12} {:<10.4} {}",
                c.id, c.memory_type, c.confidence, c.content_preview
            );
        }
    } else {
        println!("Deleted {} stale memories.", result.deleted);
    }

    Ok(())
}
