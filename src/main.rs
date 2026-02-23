mod cli;
mod config;
mod db;
mod embedding;
mod memory;
mod server;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "loci", version, about = "Cognitive memory MCP server for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the MCP server (transport from config or --transport flag)
    Serve {
        /// Override transport: "stdio" or "sse"
        #[arg(long)]
        transport: Option<String>,
    },
    /// Manage the embedding model
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Search memories from the terminal
    Search {
        /// Natural language query
        query: String,
    },
    /// Display memory statistics
    Stats {
        /// Filter stats to a specific group
        #[arg(long)]
        group: Option<String>,
    },
    /// Inspect a memory by ID
    Inspect {
        /// Memory ID to inspect
        id: String,
    },
    /// Export all memories as JSON
    Export,
    /// Import memories from a JSON file
    Import {
        /// Path to JSON file
        file: PathBuf,
    },
    /// Delete all memories (requires confirmation)
    Reset,
    /// Run maintenance compaction (decay + compact + promote)
    Compact,
    /// Clean up stale low-confidence memories
    Cleanup {
        /// Preview what would be deleted without actually deleting
        #[arg(long)]
        dry_run: bool,
    },
    /// Run database diagnostics and health check
    Doctor,
    /// Re-embed all memories with the currently configured model
    ReEmbed,
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
        Command::Serve { transport } => {
            let transport = transport.as_deref().unwrap_or(&config.server.transport);
            match transport {
                "stdio" => server::serve_stdio(config).await?,
                "sse" => server::serve_sse(config).await?,
                other => anyhow::bail!(
                    "unknown transport '{other}'. Supported: stdio, sse"
                ),
            }
        }
        Command::Model { action } => match action {
            ModelAction::Download => {
                cli::model_download(&config.embedding).await?;
            }
        },
        Command::Search { query } => {
            cli::search::search(&config, &query).await?;
        }
        Command::Stats { group } => {
            cli::stats::stats(&config, group.as_deref())?;
        }
        Command::Inspect { id } => {
            cli::inspect::inspect(&config, &id)?;
        }
        Command::Export => {
            cli::export::export(&config)?;
        }
        Command::Import { file } => {
            cli::import::import(&config, &file).await?;
        }
        Command::Reset => {
            cli::reset::reset(&config)?;
        }
        Command::Compact => {
            cli::maintenance::compact(&config).await?;
        }
        Command::Cleanup { dry_run } => {
            cli::maintenance::cleanup(&config, dry_run)?;
        }
        Command::Doctor => {
            cli::doctor::doctor(&config)?;
        }
        Command::ReEmbed => {
            cli::re_embed::re_embed(&config).await?;
        }
    }

    Ok(())
}
