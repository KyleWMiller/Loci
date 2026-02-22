use anyhow::Result;
use std::sync::Arc;

use crate::config::LociConfig;
use crate::memory::search::{SearchConfig, SearchFilter};

/// Run an interactive search from the terminal.
pub async fn search(config: &LociConfig, query: &str) -> Result<()> {
    let db_path = config.resolved_db_path();
    let conn = crate::db::open_database(&db_path)?;

    // Create embedding provider
    let provider = crate::embedding::create_provider(&config.embedding)?;
    let embedding_provider: Arc<dyn crate::embedding::EmbeddingProvider> = Arc::from(provider);

    // Embed the query
    let query_text = query.to_string();
    let ep = Arc::clone(&embedding_provider);
    let query_embedding = tokio::task::spawn_blocking(move || ep.embed(&query_text)).await??;

    let filter = SearchFilter {
        memory_type: None,
        scope: None,
        group: config.storage.default_group.clone(),
        min_confidence: 0.1,
    };

    let search_config = SearchConfig {
        max_results: config.retrieval.default_max_results,
        token_budget: config.retrieval.recall_token_budget,
        rrf_k: config.retrieval.rrf_k,
    };

    let response = crate::memory::search::recall_by_query(
        &conn,
        &query_embedding,
        query,
        &filter,
        &search_config,
    )?;

    if response.results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    println!(
        "Found {} result(s) (token estimate: ~{})\n",
        response.total_matched, response.token_estimate
    );

    for (i, result) in response.results.iter().enumerate() {
        let preview = if result.content.len() > 120 {
            format!("{}...", &result.content[..120])
        } else {
            result.content.clone()
        };

        println!(
            "  {}. [{}] {} (confidence: {:.2}, score: {:.4})",
            i + 1,
            result.memory_type,
            result.id,
            result.confidence,
            result.score,
        );
        println!("     {}", preview);
        println!();
    }

    Ok(())
}
