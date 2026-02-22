use anyhow::Result;

use crate::config::LociConfig;

/// Display memory statistics in the terminal.
pub fn stats(config: &LociConfig, group: Option<&str>) -> Result<()> {
    let db_path = config.resolved_db_path();
    let conn = crate::db::open_database(&db_path)?;

    let response = crate::memory::stats::memory_stats(&conn, group, Some(&db_path))?;

    println!("Memory Statistics");
    println!("{}", "=".repeat(40));
    println!("  Total memories:      {}", response.total_memories);
    println!("  Active:              {}", response.active_memories);
    println!("  Superseded:          {}", response.superseded_memories);
    println!();

    println!("By Type:");
    for t in &["episodic", "semantic", "procedural", "entity"] {
        let count = response.by_type.get(*t).copied().unwrap_or(0);
        println!("  {:<12} {}", t, count);
    }
    println!();

    println!("By Scope:");
    for s in &["global", "group"] {
        let count = response.by_scope.get(*s).copied().unwrap_or(0);
        println!("  {:<12} {}", s, count);
    }
    println!();

    println!("Entity relations:      {}", response.entity_relations);
    println!("Database size:         {} bytes", response.db_size_bytes);

    if let Some(ref oldest) = response.oldest_memory {
        println!("Oldest memory:         {oldest}");
    }
    if let Some(ref newest) = response.newest_memory {
        println!("Newest memory:         {newest}");
    }

    Ok(())
}
