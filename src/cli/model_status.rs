//! CLI `model status` command — check embedding model download and configuration state.

use anyhow::Result;

use crate::config::LociConfig;

/// Check the embedding model status: file presence, sizes, and DB model match.
pub fn model_status(config: &LociConfig) -> Result<()> {
    let cache_dir = crate::config::expand_tilde(&config.embedding.cache_dir);
    let model_path = cache_dir.join("model.onnx");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    println!("Loci Model Status");
    println!("==================");
    println!();
    println!("Configured model:  {}", config.embedding.model);
    println!("Provider:          {}", config.embedding.provider);
    println!("Cache directory:   {}", cache_dir.display());
    println!();

    // Check model file
    let model_ok = if model_path.exists() {
        let size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        println!("model.onnx:        {} ({})", format_bytes(size), model_path.display());
        true
    } else {
        println!("model.onnx:        MISSING");
        false
    };

    // Check tokenizer file
    let tokenizer_ok = if tokenizer_path.exists() {
        let size = std::fs::metadata(&tokenizer_path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!("tokenizer.json:    {} ({})", format_bytes(size), tokenizer_path.display());
        true
    } else {
        println!("tokenizer.json:    MISSING");
        false
    };

    println!();

    // Check DB model tracking
    let db_path = config.resolved_db_path();
    if db_path.exists() {
        if let Ok(conn) = crate::db::open_database(&db_path) {
            let stored = crate::db::migrations::get_embedding_model(&conn)
                .unwrap_or(None);
            println!("DB stored model:   {}", stored.as_deref().unwrap_or("(not set)"));
            if let Some(ref stored_model) = stored {
                if stored_model != &config.embedding.model {
                    println!("  WARNING: config model '{}' != stored model '{}'", config.embedding.model, stored_model);
                    println!("  Run `loci re-embed` to re-embed all memories with the configured model.");
                }
            }
        }
    } else {
        println!("Database:          not initialized yet");
    }

    println!();

    // Overall status
    if model_ok && tokenizer_ok {
        println!("Status:            READY");
    } else {
        println!("Status:            NOT READY");
        println!("Run `loci model download` to fetch missing files.");
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
