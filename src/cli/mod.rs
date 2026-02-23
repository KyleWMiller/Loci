pub mod doctor;
pub mod export;
pub mod import;
pub mod inspect;
pub mod maintenance;
pub mod re_embed;
pub mod reset;
pub mod search;
pub mod stats;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

const MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// Download the ONNX embedding model and tokenizer to the cache directory.
pub async fn model_download(config: &crate::config::EmbeddingConfig) -> Result<()> {
    let cache_dir = crate::config::expand_tilde(&config.cache_dir);
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create cache dir: {}", cache_dir.display()))?;

    let model_path = cache_dir.join("model.onnx");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    if model_path.exists() {
        println!("Model already exists at {}", model_path.display());
    } else {
        println!("Downloading model.onnx (~90MB)...");
        download_file(MODEL_URL, &model_path).await?;
        println!("Model saved to {}", model_path.display());
    }

    if tokenizer_path.exists() {
        println!("Tokenizer already exists at {}", tokenizer_path.display());
    } else {
        println!("Downloading tokenizer.json...");
        download_file(TOKENIZER_URL, &tokenizer_path).await?;
        println!("Tokenizer saved to {}", tokenizer_path.display());
    }

    println!("Model download complete. Ready for use.");
    Ok(())
}

/// Download a file from a URL with progress bar. Uses atomic write (tmp + rename).
async fn download_file(url: &str, dest: &PathBuf) -> Result<()> {
    let response = reqwest::get(url)
        .await
        .with_context(|| format!("HTTP request failed for {url}"))?;

    anyhow::ensure!(
        response.status().is_success(),
        "download failed with HTTP {}",
        response.status()
    );

    let total_size = response.content_length();
    let pb = if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")
                .expect("valid template")
                .progress_chars("##-"),
        );
        pb
    } else {
        ProgressBar::new_spinner()
    };

    let tmp_path = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("failed to create temp file: {}", tmp_path.display()))?;

    let bytes = response.bytes().await.context("error reading response")?;
    pb.inc(bytes.len() as u64);
    file.write_all(&bytes)
        .await
        .context("error writing to file")?;

    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp_path, dest)
        .await
        .context("failed to rename temp file")?;

    pb.finish_and_clear();
    Ok(())
}
