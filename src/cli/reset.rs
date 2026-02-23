//! CLI `reset` command — delete all memories after user confirmation.

use anyhow::{bail, Result};
use std::io::Write;

use crate::config::LociConfig;

/// Delete all memories after user confirmation.
pub fn reset(config: &LociConfig) -> Result<()> {
    let db_path = config.resolved_db_path();

    println!("WARNING: This will permanently delete ALL memories, relations, and audit logs.");
    println!("Database: {}", db_path.display());
    print!("\nType YES to confirm: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim() != "YES" {
        bail!("reset cancelled");
    }

    let conn = crate::db::open_database(&db_path)?;

    // Drop all data — order matters for FK constraints
    conn.execute_batch(
        "DELETE FROM entity_relations;
         DELETE FROM memory_log;
         DELETE FROM memories_fts;
         DELETE FROM memories_vec;
         DELETE FROM memories;",
    )?;

    println!("All memories deleted. Database reset complete.");
    Ok(())
}
