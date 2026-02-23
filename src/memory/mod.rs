//! Core memory engine — storage, search, relations, and maintenance.
//!
//! This module contains the write path ([`store`]), read path ([`search`]),
//! entity graph ([`relations`]), deletion ([`forget`]), statistics ([`stats`]),
//! and lifecycle management ([`maintenance`]). Type definitions live in [`types`].

pub mod forget;
pub mod maintenance;
pub mod relations;
pub mod search;
pub mod stats;
pub mod store;
pub mod types;

/// Convert an f32 embedding slice to raw bytes for sqlite-vec.
pub fn embedding_to_bytes(embedding: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            embedding.as_ptr() as *const u8,
            embedding.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Convert a cosine similarity threshold to L2 distance threshold.
///
/// sqlite-vec defaults to L2 distance. For L2-normalized vectors:
///   L2_dist = sqrt(2 * (1 - cosine_similarity))
pub fn cosine_threshold_to_l2(cosine_threshold: f64) -> f64 {
    (2.0 * (1.0 - cosine_threshold)).sqrt()
}
