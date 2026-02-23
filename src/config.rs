use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct LociConfig {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub retrieval: RetrievalConfig,
    pub maintenance: MaintenanceConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ServerConfig {
    pub transport: String,
    pub log_level: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct StorageConfig {
    pub db_path: String,
    pub default_group: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub cache_dir: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct RetrievalConfig {
    pub default_max_results: usize,
    pub preload_token_budget: usize,
    pub recall_token_budget: usize,
    pub rrf_k: usize,
    pub dedup_threshold: f64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct MaintenanceConfig {
    pub enabled: bool,
    pub interval_days: u64,
    pub episodic_decay_factor: f64,
    pub semantic_decay_factor: f64,
    pub compaction_age_days: u64,
    pub compaction_min_group_size: usize,
    pub promotion_threshold: usize,
    pub promotion_similarity: f64,
    pub cleanup_confidence_floor: f64,
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
        if let Ok(val) = std::env::var("LOCI_DB") {
            self.storage.db_path = val;
        }
        if let Ok(val) = std::env::var("LOCI_GROUP") {
            self.storage.default_group = val;
        }
        if let Ok(val) = std::env::var("LOCI_LOG_LEVEL") {
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
        std::env::set_var("LOCI_DB", "/tmp/override.db");
        std::env::set_var("LOCI_GROUP", "env-group");
        std::env::set_var("LOCI_LOG_LEVEL", "trace");

        config.apply_env_overrides();

        assert_eq!(config.storage.db_path, "/tmp/override.db");
        assert_eq!(config.storage.default_group, "env-group");
        assert_eq!(config.server.log_level, "trace");

        // Clean up
        std::env::remove_var("LOCI_DB");
        std::env::remove_var("LOCI_GROUP");
        std::env::remove_var("LOCI_LOG_LEVEL");
    }
}
