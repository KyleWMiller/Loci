pub mod forget;
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
