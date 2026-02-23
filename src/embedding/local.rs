//! Local ONNX Runtime embedding provider.
//!
//! Implements [`EmbeddingProvider`] using the all-MiniLM-L6-v2
//! model via `ort`. Handles tokenization, inference, mean pooling, and L2 normalization.

use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::{EmbeddingProvider, EMBEDDING_DIM};
use crate::config::EmbeddingConfig;

/// Maximum sequence length for all-MiniLM-L6-v2 (trained at 256).
const MAX_SEQ_LEN: usize = 256;

/// Local ONNX-based embedding provider using all-MiniLM-L6-v2.
pub struct LocalEmbeddingProvider {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

// Safety: Tokenizer is Send+Sync. Session is behind a Mutex.
// The Mutex guarantees exclusive access during run().
unsafe impl Send for LocalEmbeddingProvider {}
unsafe impl Sync for LocalEmbeddingProvider {}

impl LocalEmbeddingProvider {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let cache_dir = crate::config::expand_tilde(&config.cache_dir);
        let model_path = cache_dir.join("model.onnx");
        let tokenizer_path = cache_dir.join("tokenizer.json");

        anyhow::ensure!(
            model_path.exists(),
            "ONNX model not found at {}. Run `loci model download` first.",
            model_path.display()
        );
        anyhow::ensure!(
            tokenizer_path.exists(),
            "Tokenizer not found at {}. Run `loci model download` first.",
            tokenizer_path.display()
        );

        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)
            .context("failed to load ONNX model")?;

        tracing::info!(model = %model_path.display(), "ONNX model loaded");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("failed to set truncation: {e}"))?;

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        tracing::info!(tokenizer = %tokenizer_path.display(), "tokenizer loaded");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

impl EmbeddingProvider for LocalEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text])?;
        Ok(results.into_iter().next().expect("batch had one input"))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Step 2: Build flat input tensors as i64
        let mut input_ids_flat = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            for &id in encoding.get_ids() {
                input_ids_flat.push(id as i64);
            }
            for &mask in encoding.get_attention_mask() {
                attention_mask_flat.push(mask as i64);
            }
        }

        let shape = vec![batch_size as i64, seq_len as i64];
        let input_ids_tensor =
            Tensor::from_array((shape.clone(), input_ids_flat.into_boxed_slice()))?;
        let attention_mask_tensor =
            Tensor::from_array((shape.clone(), attention_mask_flat.clone().into_boxed_slice()))?;
        // token_type_ids: all zeros (single sentence, no segment B)
        let token_type_ids = vec![0i64; batch_size * seq_len];
        let token_type_ids_tensor =
            Tensor::from_array((shape, token_type_ids.into_boxed_slice()))?;

        // Step 3: Run ONNX inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        })?;

        // Step 4: Extract token embeddings — shape [batch, seq_len, 384]
        // The output name varies by ONNX export. Try common names, fall back to index 0.
        let token_emb_value = outputs
            .get("token_embeddings")
            .or_else(|| outputs.get("last_hidden_state"))
            .unwrap_or_else(|| &outputs[0]);

        let (shape, data) = token_emb_value
            .try_extract_tensor::<f32>()
            .context("failed to extract token_embeddings tensor")?;

        let dims: &[i64] = &shape;
        anyhow::ensure!(
            dims.len() == 3 && dims[2] == EMBEDDING_DIM as i64,
            "unexpected token_embeddings shape: {dims:?}, expected [batch, seq, {EMBEDDING_DIM}]"
        );
        let hidden_dim = dims[2] as usize;
        let actual_seq_len = dims[1] as usize;

        // Step 5: Mean pooling with attention mask
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut sum = vec![0.0f32; hidden_dim];
            let mut count = 0.0f32;

            for s in 0..actual_seq_len {
                let mask = attention_mask_flat[b * seq_len + s] as f32;
                if mask > 0.0 {
                    let offset = (b * actual_seq_len + s) * hidden_dim;
                    for d in 0..hidden_dim {
                        sum[d] += data[offset + d] * mask;
                    }
                    count += mask;
                }
            }

            if count > 0.0 {
                for d in 0..hidden_dim {
                    sum[d] /= count;
                }
            }

            // Step 6: L2 normalize
            results.push(l2_normalize(&sum));
        }

        Ok(results)
    }
}

/// L2-normalize a vector. Returns a zero vector if the input norm is zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = l2_normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    fn test_config() -> EmbeddingConfig {
        EmbeddingConfig {
            provider: "local".into(),
            model: "all-MiniLM-L6-v2".into(),
            cache_dir: dirs::home_dir()
                .expect("home dir")
                .join(".loci/models")
                .to_string_lossy()
                .into_owned(),
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    #[test]
    #[ignore] // Requires model files — run with: cargo test -- --ignored
    fn test_embed_produces_384_dims() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let embedding = provider.embed("Hello world").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    #[ignore]
    fn test_embed_is_l2_normalized() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let embedding = provider.embed("Test sentence for normalization").unwrap();
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    #[test]
    #[ignore]
    fn test_embed_consistency() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let emb1 = provider
            .embed("Rust is a systems programming language")
            .unwrap();
        let emb2 = provider
            .embed("Rust is a systems programming language")
            .unwrap();
        assert_eq!(emb1, emb2, "same input must produce identical output");
    }

    #[test]
    #[ignore]
    fn test_embed_batch() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let texts = vec!["First sentence", "Second sentence", "Third sentence"];
        let embeddings = provider.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), EMBEDDING_DIM);
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    #[ignore]
    fn test_similar_texts_have_high_cosine_similarity() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let emb1 = provider.embed("The cat sat on the mat").unwrap();
        let emb2 = provider.embed("A cat was sitting on a mat").unwrap();
        let emb3 = provider.embed("Quantum computing uses qubits").unwrap();

        let sim_similar = cosine_similarity(&emb1, &emb2);
        let sim_different = cosine_similarity(&emb1, &emb3);

        assert!(
            sim_similar > 0.7,
            "similar texts should have high similarity, got {sim_similar}"
        );
        assert!(
            sim_different < sim_similar,
            "different texts should have lower similarity"
        );
    }

    #[test]
    #[ignore]
    fn test_empty_batch() {
        let config = test_config();
        let provider = LocalEmbeddingProvider::new(&config).unwrap();
        let embeddings = provider.embed_batch(&[]).unwrap();
        assert!(embeddings.is_empty());
    }
}
