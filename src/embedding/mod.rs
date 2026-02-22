pub mod local;

use anyhow::Result;

/// Number of dimensions in the embedding vectors (all-MiniLM-L6-v2).
pub const EMBEDDING_DIM: usize = 384;

/// Trait for embedding text into vectors.
///
/// Implementations produce L2-normalized vectors of exactly [`EMBEDDING_DIM`] dimensions.
/// All methods are synchronous — callers in async contexts should use
/// `tokio::task::spawn_blocking`.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text string into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of text strings. Implementations may override for batched inference.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return the number of dimensions this provider produces.
    fn dimensions(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Create an embedding provider from config.
///
/// Currently only `"local"` is supported (ONNX Runtime + all-MiniLM-L6-v2).
/// Returns an error if model files are not found — run `loci model download` first.
pub fn create_provider(
    config: &crate::config::EmbeddingConfig,
) -> Result<Box<dyn EmbeddingProvider>> {
    match config.provider.as_str() {
        "local" => {
            let provider = local::LocalEmbeddingProvider::new(config)?;
            Ok(Box::new(provider))
        }
        other => anyhow::bail!("unknown embedding provider: {other}. Supported: local"),
    }
}
