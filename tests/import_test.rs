mod helpers;

use anyhow::Result;
use helpers::{test_db, test_embedding};
use loci::cli::export::{ExportData, collect_export};
use loci::cli::import::import_data;
use loci::memory::forget::forget_memory;
use loci::memory::relations::store_relation;
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};
use rusqlite::Connection;

fn store(
    conn: &mut Connection,
    content: &str,
    t: MemoryType,
    seed: u8,
    supersedes: Option<&str>,
) -> String {
    store_memory(
        conn,
        content,
        t,
        Scope::Global,
        Some("synthetic-group"),
        0.8,
        Some(&serde_json::json!({"k": "synthetic"})),
        supersedes,
        &test_embedding(seed),
        0.92,
    )
    .unwrap()
    .id
}

/// Source DB: two entities + relation, a superseded pair, and a soft-forgotten memory.
fn populated_source() -> Connection {
    let mut a = test_db();
    let e1 = store(&mut a, "synthetic entity one", MemoryType::Entity, 1, None);
    let e2 = store(&mut a, "synthetic entity two", MemoryType::Entity, 2, None);
    store_relation(&a, &e1, "knows", &e2).unwrap();
    let old = store(&mut a, "synthetic fact v1", MemoryType::Semantic, 3, None);
    store(
        &mut a,
        "synthetic fact v2",
        MemoryType::Semantic,
        4,
        Some(&old),
    );
    let gone = store(
        &mut a,
        "synthetic fact forgotten",
        MemoryType::Semantic,
        5,
        None,
    );
    forget_memory(&mut a, &gone, None, false).unwrap();
    a
}

type MemRow = (
    String,
    String,
    String,
    Option<String>,
    String,
    f64,
    String,
    String,
);

fn memory_rows(conn: &Connection) -> Vec<MemRow> {
    conn.prepare(
        "SELECT id, type, content, superseded_by, scope, confidence, created_at, updated_at \
         FROM memories ORDER BY id",
    )
    .unwrap()
    .query_map([], |r| {
        Ok((
            r.get(0)?,
            r.get(1)?,
            r.get(2)?,
            r.get(3)?,
            r.get(4)?,
            r.get(5)?,
            r.get(6)?,
            r.get(7)?,
        ))
    })
    .unwrap()
    .collect::<Result<_, _>>()
    .unwrap()
}

fn relation_rows(conn: &Connection) -> Vec<(String, String, String, String, String)> {
    conn.prepare(
        "SELECT id, subject_id, predicate, object_id, created_at FROM entity_relations ORDER BY id",
    )
    .unwrap()
    .query_map([], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
    })
    .unwrap()
    .collect::<Result<_, _>>()
    .unwrap()
}

fn count(conn: &Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

/// Deterministic fake embedder keyed by content; counts calls.
struct FakeEmbedder {
    calls: usize,
}
impl FakeEmbedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.calls += 1;
        let seed = text
            .bytes()
            .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
        Ok(test_embedding((seed % 251) as u8))
    }
}

fn roundtrip(data: &ExportData) -> ExportData {
    serde_json::from_str(&serde_json::to_string(data).unwrap()).unwrap()
}

#[test]
fn export_then_import_preserves_ids_relations_and_supersession() {
    let a = populated_source();
    let data = roundtrip(&collect_export(&a).unwrap());
    assert_eq!(data.memories.len(), 5);

    let mut b = test_db();
    let mut emb = FakeEmbedder { calls: 0 };
    let summary = import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();

    assert_eq!(summary.imported, 5);
    assert_eq!(summary.skipped, 0);
    assert_eq!(summary.relations_created, 1);
    assert_eq!(summary.supersessions_applied, 2); // superseded + forgotten
    assert_eq!(memory_rows(&a), memory_rows(&b));
    assert_eq!(relation_rows(&a), relation_rows(&b));

    // Index rows exist for the preserved IDs.
    assert_eq!(count(&b, "memories_vec"), 5);
    let fts: i64 = b
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'entity'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fts, 2);
}

#[test]
fn superseded_memories_stay_superseded_including_forgotten() {
    let a = populated_source();
    let data = collect_export(&a).unwrap();
    let mut b = test_db();
    let mut emb = FakeEmbedder { calls: 0 };
    import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();

    let active: i64 = b
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active, 3); // two entities + fact v2
    let forgotten: i64 = b
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE superseded_by = 'forgotten'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(forgotten, 1);
}

#[test]
fn importing_twice_changes_nothing_and_does_not_reembed() {
    let a = populated_source();
    let data = collect_export(&a).unwrap();
    let mut b = test_db();
    let mut emb = FakeEmbedder { calls: 0 };
    import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();

    let tables = ["memories", "memories_vec", "entity_relations", "memory_log"];
    let before: Vec<i64> = tables.iter().map(|t| count(&b, t)).collect();
    let rows_before = memory_rows(&b);
    let calls_before = emb.calls;

    let again = import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();
    assert_eq!(again.imported, 0);
    assert_eq!(again.skipped, 5);
    assert_eq!(again.relations_created, 0);
    assert_eq!(again.relations_existing, 1);
    assert_eq!(
        emb.calls, calls_before,
        "existing IDs must not be re-embedded"
    );
    let after: Vec<i64> = tables.iter().map(|t| count(&b, t)).collect();
    assert_eq!(before, after);
    assert_eq!(rows_before, memory_rows(&b));
}

#[test]
fn import_skips_locally_hard_deleted_ids() {
    let a = populated_source();
    let data = collect_export(&a).unwrap();
    let victim = data.memories[0].id.clone();
    let mut b = test_db();
    b.execute(
        "INSERT INTO tombstones (memory_id, hard, ts) VALUES (?1, 1, '2026-01-01T00:00:00Z')",
        [&victim],
    )
    .unwrap();
    let mut emb = FakeEmbedder { calls: 0 };
    let s = import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();
    assert_eq!(s.tombstoned, 1);
    assert_eq!(count(&b, "memories"), 4);
}

#[test]
fn relation_with_missing_endpoint_is_skipped_not_fatal() {
    let a = populated_source();
    let mut data = collect_export(&a).unwrap();
    data.relations[0].object_id = "0190ffff-0000-7000-8000-00000000dead".into();
    let mut b = test_db();
    let mut emb = FakeEmbedder { calls: 0 };
    let s = import_data(&mut b, &data, &mut |t| emb.embed(t)).unwrap();
    assert_eq!(s.relations_skipped, 1);
    assert_eq!(count(&b, "entity_relations"), 0);
}
