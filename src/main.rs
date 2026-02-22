mod cli;
mod config;
mod db;
mod embedding;
mod memory;
mod server;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "loci", version, about = "Cognitive memory MCP server for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the MCP server (stdio transport)
    Serve,
    /// Manage the embedding model
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download the embedding model to ~/.loci/models/
    Download,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load config (for log level)
    let config = config::LociConfig::load()?;

    // Initialize tracing with the configured log level.
    // Log to stderr so stdout stays clean for MCP JSON-RPC.
    let filter = EnvFilter::try_new(&config.server.log_level)
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Command::Serve => {
            server::serve_stdio(config).await?;
        }
        Command::Model { action } => match action {
            ModelAction::Download => {
                cli::model_download(&config.embedding).await?;
            }
        },
    }

    Ok(())
}
