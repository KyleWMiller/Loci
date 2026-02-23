//! Configuration loading and management.
//!
//! Loci reads configuration from `~/.loci/config.toml` (if present) with environment
//! variable overrides (`LOCI_DB`, `LOCI_GROUP`, `LOCI_LOG_LEVEL`). All fields have
//! sensible defaults — no configuration file is required.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::info;

/// Top-level Loci configuration, deserialized from `config.toml`.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct LociConfig {
    /// MCP server transport and logging settings.
    pub server: ServerConfig,
    /// Database path and default group.
    pub storage: StorageConfig,
    /// Embedding model and cache directory.
    pub embedding: EmbeddingConfig,
    /// Search parameters (max results, token budgets, RRF, dedup).
    pub retrieval: RetrievalConfig,
    /// Lifecycle management (decay, compaction, promotion, cleanup).
    pub maintenance: MaintenanceConfig,
}

/// MCP server transport and logging settings.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ServerConfig {
    /// Transport type: `"stdio"` (default) or `"sse"`.
    pub transport: String,
    /// Tracing log level (e.g. `"info"`, `"debug"`, `"trace"`).
    pub log_level: String,
    /// Bind address for SSE transport (default `"127.0.0.1"`).
    pub host: String,
    /// Port for SSE transport (default `8080`).
    pub port: u16,
}

/// Database path and default memory group.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct StorageConfig {
    /// Path to the SQLite database file (supports `~` expansion).
    pub db_path: String,
    /// Default `source_group` for new memories (default `"default"`).
    pub default_group: String,
}

/// Embedding model configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Provider type: `"local"` for ONNX Runtime (only option currently).
    pub provider: String,
    /// Model identifier (default `"all-MiniLM-L6-v2"`).
    pub model: String,
    /// Directory to cache model files (supports `~` expansion).
    pub cache_dir: String,
}

/// Search and deduplication parameters.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct RetrievalConfig {
    /// Maximum results returned by `recall_memory` (default 5).
    pub default_max_results: usize,
    /// Token budget for preload/summary mode (default 2000).
    pub preload_token_budget: usize,
    /// Token budget for full recall (default 4000).
    pub recall_token_budget: usize,
    /// Reciprocal Rank Fusion constant `k` (default 60).
    pub rrf_k: usize,
    /// Cosine similarity threshold for deduplication (default 0.92).
    pub dedup_threshold: f64,
}

/// Memory lifecycle management settings.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct MaintenanceConfig {
    /// Enable automatic maintenance on startup (default `false`).
    pub enabled: bool,
    /// Days between automatic maintenance runs (default 7).
    pub interval_days: u64,
    /// Per-cycle decay multiplier for episodic memories (default 0.95).
    pub episodic_decay_factor: f64,
    /// Per-cycle decay multiplier for semantic/procedural/entity memories (default 0.99).
    pub semantic_decay_factor: f64,
    /// Minimum age in days before episodic memories are eligible for compaction (default 30).
    pub compaction_age_days: u64,
    /// Minimum group size for episodic compaction (default 5).
    pub compaction_min_group_size: usize,
    /// Minimum cluster size for episodic-to-semantic promotion (default 3).
    pub promotion_threshold: usize,
    /// Cosine similarity threshold for promotion clustering (default 0.88).
    pub promotion_similarity: f64,
    /// Confidence below this floor makes a memory eligible for cleanup (default 0.05).
    pub cleanup_confidence_floor: f64,
    /// Days without access before a low-confidence memory is cleaned up (default 90).
    pub cleanup_no_access_days: u64,
}

impl Default for LociConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            retrieval: RetrievalConfig::default(),
            maintenance: MaintenanceConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            transport: "stdio".into(),
            log_level: "info".into(),
            host: "127.0.0.1".into(),
            port: 8080,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        let db_path = default_loci_dir()
            .join("memory.db")
            .to_string_lossy()
            .into_owned();
        Self {
            db_path,
            default_group: "default".into(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let cache_dir = default_loci_dir()
            .join("models")
            .to_string_lossy()
            .into_owned();
        Self {
            provider: "local".into(),
            model: "all-MiniLM-L6-v2".into(),
            cache_dir,
        }
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_max_results: 5,
            preload_token_budget: 2000,
            recall_token_budget: 4000,
            rrf_k: 60,
            dedup_threshold: 0.92,
        }
    }
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_days: 7,
            episodic_decay_factor: 0.95,
            semantic_decay_factor: 0.99,
            compaction_age_days: 30,
            compaction_min_group_size: 5,
            promotion_threshold: 3,
            promotion_similarity: 0.88,
            cleanup_confidence_floor: 0.05,
            cleanup_no_access_days: 90,
        }
    }
}

/// Returns `~/.loci/`
pub fn default_loci_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home directory must exist")
        .join(".loci")
}

/// Returns the default config file path: `~/.loci/config.toml`
pub fn default_config_path() -> PathBuf {
    default_loci_dir().join("config.toml")
}

impl LociConfig {
    /// Load config from TOML file (if it exists) then apply env var overrides.
    pub fn load() -> Result<Self> {
        Self::load_from(default_config_path())
    }

    /// Load from a specific path, then apply env var overrides.
    pub fn load_from(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut config = if path.exists() {
            let contents =
                std::fs::read_to_string(path).context("failed to read config file")?;
            toml::from_str(&contents).context("failed to parse config TOML")?
        } else {
            info!("no config file at {}, using defaults", path.display());
            LociConfig::default()
        };

        config.apply_env_overrides();
        Ok(config)
    }

    /// Apply environment variable overrides (LOCI_DB, LOCI_GROUP, LOCI_LOG_LEVEL).
    fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_with(|key| std::env::var(key));
    }

    /// Apply overrides using a custom env lookup function.
    fn apply_env_overrides_with(&mut self, env: impl Fn(&str) -> Result<String, std::env::VarError>) {
        if let Ok(val) = env("LOCI_DB") {
            self.storage.db_path = val;
        }
        if let Ok(val) = env("LOCI_GROUP") {
            self.storage.default_group = val;
        }
        if let Ok(val) = env("LOCI_LOG_LEVEL") {
            self.server.log_level = val;
        }
    }

    /// Resolve the database path, expanding `~` if needed.
    pub fn resolved_db_path(&self) -> PathBuf {
        expand_tilde(&self.storage.db_path)
    }
}

pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        dirs::home_dir()
            .expect("home directory must exist")
            .join(rest)
    } else {
        PathBuf::from(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = LociConfig::default();
        assert_eq!(config.server.transport, "stdio");
        assert_eq!(config.server.log_level, "info");
        assert_eq!(config.storage.default_group, "default");
        assert_eq!(config.retrieval.rrf_k, 60);
        assert!(config.storage.db_path.ends_with("memory.db"));
    }

    #[test]
    fn parse_toml_config() {
        let toml_str = r#"
[server]
log_level = "debug"

[storage]
db_path = "/tmp/test.db"
default_group = "myproject"

[retrieval]
default_max_results = 10
"#;
        let config: LociConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.storage.db_path, "/tmp/test.db");
        assert_eq!(config.storage.default_group, "myproject");
        assert_eq!(config.retrieval.default_max_results, 10);
        // defaults still apply for unset fields
        assert_eq!(config.retrieval.rrf_k, 60);
    }

    #[test]
    fn env_overrides_apply() {
        let mut config = LociConfig::default();
        let env = |key: &str| match key {
            "LOCI_DB" => Ok("/tmp/override.db".into()),
            "LOCI_GROUP" => Ok("env-group".into()),
            "LOCI_LOG_LEVEL" => Ok("trace".into()),
            _ => Err(std::env::VarError::NotPresent),
        };

        config.apply_env_overrides_with(env);

        assert_eq!(config.storage.db_path, "/tmp/override.db");
        assert_eq!(config.storage.default_group, "env-group");
        assert_eq!(config.server.log_level, "trace");
    }
}
