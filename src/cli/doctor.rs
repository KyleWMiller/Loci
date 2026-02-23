//! CLI `doctor` command — run database diagnostics and print a health report.

use anyhow::{Context, Result};

use crate::config::LociConfig;
use crate::db;

/// Run database diagnostics and print a health report.
pub fn doctor(config: &LociConfig) -> Result<()> {
    let db_path = config.resolved_db_path();

    if !db_path.exists() {
        println!("Database: not found at {}", db_path.display());
        println!("Run `loci serve` or `loci model download` to initialize.");
        return Ok(());
    }

    let file_size = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let conn = db::open_database(&db_path)
        .context("failed to open database (may be corrupt)")?;

    let report = db::check_database_health(&conn)
        .context("failed to run health check")?;

    println!("Loci Health Report");
    println!("==================");
    println!();
    println!("Database:          {}", db_path.display());
    println!("File size:         {}", format_bytes(file_size));
    println!("Schema version:    {}", report.schema_version);
    println!("sqlite-vec:        v{}", report.sqlite_vec_version);
    println!();
    println!("Embedding model:");
    println!("  Stored:          {}", report.embedding_model.as_deref().unwrap_or("(not set)"));
    println!("  Configured:      {}", config.embedding.model);
    if let Some(ref stored) = report.embedding_model {
        if stored != &config.embedding.model {
            println!("  WARNING: model mismatch! Run `loci re-embed` to update vectors.");
        } else {
            println!("  Status:          OK (match)");
        }
    }
    println!();
    println!("Row counts:");
    println!("  Memories:        {}", report.memory_count);
    println!("  Relations:       {}", report.relation_count);
    println!("  Audit log:       {}", report.log_count);
    println!();
    if report.integrity_ok {
        println!("Integrity check:   PASSED");
    } else {
        println!("Integrity check:   FAILED ({})", report.integrity_details);
    }

    if !report.integrity_ok {
        println!();
        println!("Recovery steps:");
        println!("  1. Restore from a backup: cp backup.db ~/.loci/memory.db");
        println!("  2. Or export from a good copy and reimport:");
        println!("     loci export > backup.json");
        println!("     loci reset && loci import backup.json");
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
