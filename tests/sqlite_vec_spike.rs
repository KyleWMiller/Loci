use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;
use std::mem;

/// Load sqlite-vec into a rusqlite connection via auto_extension.
/// Must be called before opening any connections that need vec0.
fn load_sqlite_vec() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    }
}

#[test]
fn sqlite_vec_spike_end_to_end() {
    load_sqlite_vec();

    let conn = Connection::open_in_memory().expect("open in-memory db");

    // Verify sqlite-vec is loaded
    let version: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .expect("vec_version");
    assert!(!version.is_empty(), "sqlite-vec version should be non-empty");
    println!("sqlite-vec version: {version}");

    // Create a vec0 virtual table with 384 dimensions (matching all-MiniLM-L6-v2)
    conn.execute_batch(
        "CREATE VIRTUAL TABLE test_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[384]
        );",
    )
    .expect("create vec0 table");

    // Create a test embedding (384 floats)
    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let embedding_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4)
    };

    // Insert a test vector
    conn.execute(
        "INSERT INTO test_vec (id, embedding) VALUES (?, ?)",
        rusqlite::params!["test-id-1", embedding_bytes],
    )
    .expect("insert vector");

    // Insert a second vector (slightly different)
    let embedding2: Vec<f32> = (0..384).map(|i| ((i + 1) as f32) / 384.0).collect();
    let embedding2_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(embedding2.as_ptr() as *const u8, embedding2.len() * 4)
    };
    conn.execute(
        "INSERT INTO test_vec (id, embedding) VALUES (?, ?)",
        rusqlite::params!["test-id-2", embedding2_bytes],
    )
    .expect("insert vector 2");

    // KNN query — find nearest neighbor to the first embedding
    let query_bytes = embedding_bytes;
    let mut stmt = conn
        .prepare(
            "SELECT id, distance
             FROM test_vec
             WHERE embedding MATCH ?
             ORDER BY distance
             LIMIT 5",
        )
        .expect("prepare KNN query");

    let results: Vec<(String, f64)> = stmt
        .query_map(rusqlite::params![query_bytes], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .expect("execute KNN query")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect results");

    assert!(!results.is_empty(), "KNN query should return results");
    assert_eq!(results[0].0, "test-id-1", "closest match should be itself");
    assert!(
        results[0].1 < 0.001,
        "distance to self should be ~0, got {}",
        results[0].1
    );

    println!("KNN results: {results:?}");
    println!("sqlite-vec spike test PASSED");
}
