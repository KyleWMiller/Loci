//! MCP server initialization for stdio and SSE transports.
//!
//! Provides [`serve_stdio`] and [`serve_sse`] entry points that wire up the database,
//! embedding provider, and MCP tool handler into a running server.

use crate::config::LociConfig;
use crate::db;
use crate::embedding;
use crate::tools::LociTools;
use anyhow::Result;
use rmcp::ServiceExt;
use std::sync::{Arc, Mutex};

/// Shared setup: open DB, create embedding provider, check model version.
/// Returns (db, embedding, config) wrapped in Arc for sharing.
fn setup_shared_state(
    config: LociConfig,
) -> Result<(
    Arc<Mutex<rusqlite::Connection>>,
    Arc<dyn embedding::EmbeddingProvider>,
    Arc<LociConfig>,
)> {
    let db_path = config.resolved_db_path();
    let conn = db::open_database(&db_path)?;
    tracing::info!(db = %db_path.display(), "database ready");

    // Check for embedding model mismatch
    if let Ok(Some(stored_model)) = db::migrations::get_embedding_model(&conn) {
        if stored_model != config.embedding.model {
            tracing::warn!(
                stored = %stored_model,
                configured = %config.embedding.model,
                "embedding model changed — run `loci re-embed` to update all vectors"
            );
        }
    }

    let db = Arc::new(Mutex::new(conn));

    let provider = embedding::create_provider(&config.embedding)?;
    let embedding: Arc<dyn embedding::EmbeddingProvider> = Arc::from(provider);
    tracing::info!("embedding provider ready");

    let config = Arc::new(config);

    Ok((db, embedding, config))
}

/// Start the MCP server over stdio transport.
pub async fn serve_stdio(config: LociConfig) -> Result<()> {
    tracing::info!("starting Loci MCP server on stdio");

    let (db, embedding, config) = setup_shared_state(config)?;

    let tools = LociTools::new(db, embedding, config);
    let transport = rmcp::transport::stdio();

    let server = tools.serve(transport).await?;
    tracing::info!("MCP server running — waiting for client");

    server.waiting().await?;
    tracing::info!("MCP server shut down");

    Ok(())
}

/// Start the MCP server over Streamable HTTP (SSE) transport.
pub async fn serve_sse(config: LociConfig) -> Result<()> {
    let host = config.server.host.clone();
    let port = config.server.port;
    let bind_addr = format!("{host}:{port}");

    tracing::info!(addr = %bind_addr, "starting Loci MCP server on SSE/HTTP");

    let (db, embedding, config) = setup_shared_state(config)?;

    let service = rmcp::transport::streamable_http_server::StreamableHttpService::new(
        move || Ok(LociTools::new(db.clone(), embedding.clone(), config.clone())),
        rmcp::transport::streamable_http_server::session::local::LocalSessionManager::default()
            .into(),
        Default::default(),
    );

    let router = axum::Router::new().nest_service("/mcp", service);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!(addr = %bind_addr, "MCP server listening at http://{bind_addr}/mcp");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for ctrl-c");
            tracing::info!("shutting down SSE server");
        })
        .await?;

    Ok(())
}
