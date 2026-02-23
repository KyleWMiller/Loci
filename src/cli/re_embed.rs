//! CLI `re-embed` command — regenerate all embeddings with the current model.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;

use crate::config::LociConfig;
use crate::db;
use crate::embedding;
use crate::memory::embedding_to_bytes;

/// Re-embed all active memories with the currently configured model.
pub async fn re_embed(config: &LociConfig) -> Result<()> {
    let db_path = config.resolved_db_path();
    let conn = db::open_database(&db_path)
        .context("failed to open database")?;

    // Load embedding provider
    let provider: Arc<dyn embedding::EmbeddingProvider> =
        Arc::from(embedding::create_provider(&config.embedding)
            .context("failed to create embedding provider")?);

    // Fetch all active memories
    let memories: Vec<(String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT id, content FROM memories WHERE superseded_by IS NULL"
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        rows
    };

    let total = memories.len();
    if total == 0 {
        println!("No active memories to re-embed.");
        return Ok(());
    }

    println!("Re-embedding {total} memories with model '{}'...", config.embedding.model);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .expect("valid template")
            .progress_chars("##-"),
    );

    // Process in batches of 32
    const BATCH_SIZE: usize = 32;
    for chunk in memories.chunks(BATCH_SIZE) {
        let texts: Vec<String> = chunk.iter().map(|(_, content)| content.clone()).collect();
        let provider = Arc::clone(&provider);

        let embeddings = tokio::task::spawn_blocking(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            provider.embed_batch(&text_refs)
        })
        .await?
        .context("embedding batch failed")?;

        for ((id, _), emb) in chunk.iter().zip(embeddings.iter()) {
            let bytes = embedding_to_bytes(emb);
            // Delete old vector and insert new one
            conn.execute("DELETE FROM memories_vec WHERE id = ?1", [id])?;
            conn.execute(
                "INSERT INTO memories_vec (id, embedding) VALUES (?1, ?2)",
                rusqlite::params![id, bytes],
            )?;
        }

        pb.inc(chunk.len() as u64);
    }

    pb.finish_and_clear();

    // Update stored model identifier
    db::migrations::set_embedding_model(&conn, &config.embedding.model)?;

    println!("Re-embedded {total} memories with model '{}'.", config.embedding.model);
    Ok(())
}
