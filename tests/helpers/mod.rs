#![allow(dead_code)]

use loci::db;
use loci::memory::embedding_to_bytes;
use rusqlite::Connection;

/// Open a fresh in-memory database with schema and migrations applied.
pub fn test_db() -> Connection {
    db::load_sqlite_vec();
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    db::schema::init_schema(&conn).unwrap();
    db::migrations::run_migrations(&conn).unwrap();
    conn
}

/// Generate a deterministic 384-dim embedding with a spike at position `seed`.
/// Each seed produces a distinct, orthogonal-ish vector.
pub fn test_embedding(seed: u8) -> Vec<f32> {
    let mut v = vec![0.0f32; 384];
    v[seed as usize % 384] = 1.0;
    v
}

/// Generate an embedding similar to `base` with small perturbation.
/// The result will have high cosine similarity to `base`.
pub fn similar_embedding(base: &[f32]) -> Vec<f32> {
    let mut v = base.to_vec();
    // Add small noise to a few dimensions to create near-duplicate
    for i in 0..5 {
        v[(i * 37) % 384] += 0.05;
    }
    // L2 normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Insert a test memory directly via the store module. Returns the memory ID.
pub fn insert_memory(
    conn: &mut Connection,
    content: &str,
    memory_type: loci::memory::types::MemoryType,
    scope: loci::memory::types::Scope,
    group: &str,
    confidence: f64,
    embedding: &[f32],
) -> String {
    loci::memory::store::store_memory(
        conn,
        content,
        memory_type,
        scope,
        Some(group),
        confidence,
        None,
        None,
        embedding,
        0.92, // dedup threshold
    )
    .unwrap()
    .id
}

/// Convert embedding to raw bytes (convenience wrapper for tests).
pub fn emb_bytes(embedding: &[f32]) -> &[u8] {
    embedding_to_bytes(embedding)
}
