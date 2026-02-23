//! CLI `inspect` command — display full details for a single memory.

use anyhow::Result;

use crate::config::LociConfig;

/// Inspect a single memory by ID and display full details.
pub fn inspect(config: &LociConfig, id: &str) -> Result<()> {
    let db_path = config.resolved_db_path();
    let conn = crate::db::open_database(&db_path)?;

    let response = crate::memory::search::inspect_memory(&conn, id, true, true)?;

    let m = &response.memory;
    println!("Memory: {}", m.id);
    println!("{}", "=".repeat(50));
    println!("  Type:           {}", m.memory_type);
    println!("  Confidence:     {:.2}", m.confidence);
    println!("  Access count:   {}", m.access_count);
    if let Some(ref la) = m.last_accessed {
        println!("  Last accessed:  {la}");
    }
    println!("  Created:        {}", m.created_at);
    println!("  Updated:        {}", m.updated_at);
    if let Some(ref sb) = m.superseded_by {
        println!("  Superseded by:  {sb}");
    }
    if let Some(ref meta) = m.metadata {
        println!("  Metadata:       {}", serde_json::to_string_pretty(meta)?);
    }
    println!();
    println!("Content:");
    println!("  {}", m.content);

    if let Some(ref relations) = response.relations {
        if !relations.is_empty() {
            println!();
            println!("Relations:");
            for rel in relations {
                println!(
                    "  --[{}]--> {} ({}: {})",
                    rel.predicate, rel.object.id, rel.object.memory_type, rel.object.preview,
                );
            }
        }
    }

    if let Some(ref log) = response.log {
        if !log.is_empty() {
            println!();
            println!("Audit Log:");
            for entry in log {
                let details = entry
                    .details
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default();
                println!("  {} [{}] {}", entry.created_at, entry.operation, details);
            }
        }
    }

    Ok(())
}
