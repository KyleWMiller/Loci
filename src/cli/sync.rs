//! CLI `sync` command — replicate with other machines through a shared directory.

use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::config::{LociConfig, expand_tilde};

/// Apply other machines' ops from `dir`, then publish ours.
///
/// The embedding model is only loaded if a remote memory actually has to be inserted.
pub fn sync(config: &LociConfig, dir: Option<&Path>) -> Result<()> {
    let dir = match (dir, config.sync.dir.as_deref()) {
        (Some(d), _) => d.to_path_buf(),
        (None, Some(d)) => expand_tilde(d),
        (None, None) => bail!("no sync directory: pass one or set [sync] dir in config.toml"),
    };

    let mut conn = crate::db::open_database(config.resolved_db_path())?;
    if let Some(id) = config.sync.machine_id.as_deref() {
        crate::sync::set_machine_id_override(&conn, id)?;
    }
    let machine_id = crate::db::migrations::get_machine_id(&conn)?;

    let mut provider: Option<Box<dyn crate::embedding::EmbeddingProvider>> = None;
    let mut embed = |text: &str| -> Result<Vec<f32>> {
        if provider.is_none() {
            provider = Some(
                crate::embedding::create_provider(&config.embedding)
                    .context("failed to load the embedding model (run `loci model download`)")?,
            );
        }
        match provider.as_ref() {
            Some(p) => p.embed(text),
            None => bail!("embedding provider unavailable"),
        }
    };

    let s = crate::sync::run_sync(&mut conn, &dir, &mut embed)?;

    println!(
        "Sync complete (machine {machine_id}, dir {}):",
        dir.display()
    );
    println!("  Ops written:     {}", s.written);
    println!("  Ops applied:     {}", s.applied);
    println!(
        "  Ops skipped:     {} (already known or deleted)",
        s.skipped
    );
    println!(
        "  Ops deferred:    {} (waiting for a dependency)",
        s.deferred
    );
    println!("  Forks recorded:  {}", s.forks);
    println!("  Tombstones:      {}", s.tombstones);
    if s.discarded > 0 {
        println!("  Puts discarded:  {} (hard-deleted IDs)", s.discarded);
    }
    if s.scrubbed > 0 {
        println!("  Lines scrubbed:  {} (hard-deleted content)", s.scrubbed);
    }
    if s.malformed > 0 {
        println!("  Malformed lines: {}", s.malformed);
    }
    Ok(())
}
