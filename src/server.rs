use crate::config::LociConfig;
use crate::db;
use crate::embedding;
use crate::tools::LociTools;
use anyhow::Result;
use rmcp::ServiceExt;
use std::sync::{Arc, Mutex};

/// Start the MCP server over stdio transport.
pub async fn serve_stdio(config: LociConfig) -> Result<()> {
    tracing::info!("starting Loci MCP server on stdio");

    // Open database
    let db_path = config.resolved_db_path();
    let conn = db::open_database(&db_path)?;
    tracing::info!(db = %db_path.display(), "database ready");
    let db = Arc::new(Mutex::new(conn));

    // Create embedding provider
    let provider = embedding::create_provider(&config.embedding)?;
    let embedding: Arc<dyn embedding::EmbeddingProvider> = Arc::from(provider);
    tracing::info!("embedding provider ready");

    let config = Arc::new(config);

    let tools = LociTools::new(db, embedding, config);
    let transport = rmcp::transport::stdio();

    let server = tools.serve(transport).await?;
    tracing::info!("MCP server running — waiting for client");

    server.waiting().await?;
    tracing::info!("MCP server shut down");

    Ok(())
}
