#![allow(clippy::field_reassign_with_default)]
//! Step 4: every local mutation emits exactly the expected replication op,
//! in the same transaction, and hard deletes never leave content behind.

mod helpers;

use helpers::{test_db, test_embedding};
use loci::config::{LociConfig, MaintenanceConfig};
use loci::db;
use loci::embedding::EmbeddingProvider;
use loci::memory::maintenance::compact_episodic;
use loci::memory::relations::store_relation;
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};
use loci::sync::ops::{Op, OpEnvelope};
use loci::tools::LociTools;
use loci::tools::forget_memory::ForgetMemoryParams;
use loci::tools::store_memory::StoreMemoryParams;
use rmcp::handler::server::wrapper::Parameters;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

struct FakeEmbedding;
impl EmbeddingProvider for FakeEmbedding {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut v = vec![0.0f32; 384];
        v[text.bytes().map(|b| b as usize).sum::<usize>() % 384] = 1.0;
        Ok(v)
    }
}

fn tools() -> (LociTools, Arc<Mutex<Connection>>) {
    db::load_sqlite_vec();
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    db::schema::init_schema(&conn).unwrap();
    db::migrations::run_migrations(&conn).unwrap();
    let db = Arc::new(Mutex::new(conn));
    let t = LociTools::new(
        Arc::clone(&db),
        Arc::new(FakeEmbedding),
        Arc::new(LociConfig::default()),
    );
    (t, db)
}

fn ops(conn: &Connection) -> Vec<OpEnvelope> {
    conn.prepare("SELECT body FROM op_log ORDER BY seq")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|b| serde_json::from_str(&b.unwrap()).unwrap())
        .collect()
}

fn store_params(content: &str, supersedes: Option<String>) -> StoreMemoryParams {
    StoreMemoryParams {
        content: content.into(),
        r#type: "semantic".into(),
        group: None,
        scope: None,
        confidence: None,
        metadata: Some(serde_json::json!({"k": "synthetic"})),
        supersedes,
    }
}

async fn store(t: &LociTools, content: &str, supersedes: Option<String>) -> String {
    let out = t
        .store_memory(Parameters(store_params(content, supersedes)))
        .await
        .unwrap();
    serde_json::from_str::<serde_json::Value>(&out).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn store_memory_tool_emits_one_put_with_stored_state() {
    let (t, db) = tools();
    let id = store(&t, "synthetic alpha content", None).await;

    let conn = db.lock().unwrap();
    let machine = db::migrations::get_machine_id(&conn).unwrap();
    let all = ops(&conn);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].v, 1);
    assert_eq!(all[0].origin, machine);
    assert_eq!(
        uuid::Uuid::parse_str(&all[0].op_id)
            .unwrap()
            .get_version_num(),
        7
    );
    match &all[0].op {
        Op::MemoryPut { memory } => {
            assert_eq!(memory.id, id);
            assert_eq!(memory.content, "synthetic alpha content");
            assert_eq!(memory.memory_type, "semantic");
            assert_eq!(memory.metadata, Some(serde_json::json!({"k": "synthetic"})));
            let created: String = conn
                .query_row(
                    "SELECT created_at FROM memories WHERE id = ?1",
                    [&id],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(memory.created_at, created);
        }
        other => panic!("expected put, got {other:?}"),
    }
}

#[tokio::test]
async fn dedup_bump_emits_nothing() {
    let (t, db) = tools();
    let id1 = store(&t, "synthetic same text", None).await;
    let id2 = store(&t, "synthetic same text", None).await; // identical embedding -> dedup
    assert_eq!(id1, id2);
    assert_eq!(ops(&db.lock().unwrap()).len(), 1);
}

#[tokio::test]
async fn supersedes_emits_put_then_supersede() {
    let (t, db) = tools();
    let old = store(&t, "synthetic v1", None).await;
    let new = store(&t, "synthetic completely different v2", Some(old.clone())).await;
    let all = ops(&db.lock().unwrap());
    assert_eq!(all.len(), 3);
    assert!(matches!(&all[1].op, Op::MemoryPut { memory } if memory.id == new));
    assert_eq!(
        all[2].op,
        Op::MemorySupersede {
            old_id: old,
            new_id: new
        }
    );
}

#[tokio::test]
async fn soft_forget_emits_id_only_op_without_reason() {
    let (t, db) = tools();
    let id = store(&t, "synthetic soft target", None).await;
    t.forget_memory(Parameters(ForgetMemoryParams {
        memory_id: id.clone(),
        reason: Some("synthetic-secret-reason".into()),
        hard_delete: None,
    }))
    .await
    .unwrap();

    let conn = db.lock().unwrap();
    let all = ops(&conn);
    assert_eq!(all.len(), 2);
    assert_eq!(
        all[1].op,
        Op::MemoryForget {
            id: id.clone(),
            hard: false
        }
    );
    let raw: String = conn
        .query_row(
            "SELECT body FROM op_log WHERE kind = 'memory.forget'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!raw.contains("synthetic-secret-reason"));
    // Soft delete keeps the put (the memory still exists, as forgotten).
    assert!(matches!(&all[0].op, Op::MemoryPut { .. }));
    let hard: i64 = conn
        .query_row(
            "SELECT hard FROM tombstones WHERE memory_id = ?1",
            [&id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(hard, 0);
}

#[tokio::test]
async fn hard_forget_emits_id_only_op_and_scrubs_content_from_log() {
    let (t, db) = tools();
    let id = store(&t, "synthetic unique-needle-4711 body", None).await;
    t.forget_memory(Parameters(ForgetMemoryParams {
        memory_id: id.clone(),
        reason: None,
        hard_delete: Some(true),
    }))
    .await
    .unwrap();

    let conn = db.lock().unwrap();
    let all = ops(&conn);
    // Unexported put is removed outright; only the forget remains.
    assert_eq!(all.len(), 1);
    assert_eq!(
        all[0].op,
        Op::MemoryForget {
            id: id.clone(),
            hard: true
        }
    );
    for table_col in [
        ("op_log", "body"),
        ("memory_log", "details"),
        ("memories", "content"),
        ("deferred_ops", "body"),
    ] {
        let n: i64 = conn
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM {} WHERE {} LIKE '%needle-4711%'",
                    table_col.0, table_col.1
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "content leaked into {}", table_col.0);
    }
    let hard: i64 = conn
        .query_row(
            "SELECT hard FROM tombstones WHERE memory_id = ?1",
            [&id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(hard, 1);
}

#[test]
fn hard_forget_after_export_turns_put_into_pending_scrub_noop() {
    let mut conn = test_db();
    let id = store_memory(
        &mut conn,
        "synthetic exported needle-9 text",
        MemoryType::Semantic,
        Scope::Global,
        None,
        1.0,
        None,
        None,
        &test_embedding(1),
        0.92,
    )
    .unwrap()
    .id;
    conn.execute(
        "UPDATE op_log SET exported = 1, file_offset = 0, file_len = 500",
        [],
    )
    .unwrap();

    loci::memory::forget::forget_memory(&mut conn, &id, None, true).unwrap();

    let (kind, body, pending): (String, String, i64) = conn
        .query_row(
            "SELECT kind, body, scrub_pending FROM op_log WHERE seq = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(kind, "noop");
    assert_eq!(pending, 1);
    assert!(!body.contains("needle-9"));
}

#[test]
fn store_relation_emits_put_and_dedup_emits_nothing() {
    let mut conn = test_db();
    let mk = |conn: &mut Connection, c: &str, s: u8| {
        store_memory(
            conn,
            c,
            MemoryType::Entity,
            Scope::Global,
            None,
            1.0,
            None,
            None,
            &test_embedding(s),
            0.92,
        )
        .unwrap()
        .id
    };
    let a = mk(&mut conn, "synthetic entity a", 1);
    let b = mk(&mut conn, "synthetic entity b", 2);
    let r = store_relation(&conn, &a, "knows", &b).unwrap();
    let again = store_relation(&conn, &a, "knows", &b).unwrap();
    assert!(again.deduplicated);

    let all = ops(&conn);
    assert_eq!(all.len(), 3); // 2 puts + 1 relation.put
    match &all[2].op {
        Op::RelationPut { relation } => {
            assert_eq!(relation.id, r.id);
            assert_eq!(
                (relation.subject_id.as_str(), relation.object_id.as_str()),
                (a.as_str(), b.as_str())
            );
            assert_eq!(relation.predicate, "knows");
        }
        other => panic!("expected relation.put, got {other:?}"),
    }
}

#[test]
fn compaction_emits_summary_put_and_one_supersede_per_original() {
    let mut conn = test_db();
    let mut ids = vec![];
    for i in 0..3u8 {
        let id = store_memory(
            &mut conn,
            &format!("synthetic episode {i}"),
            MemoryType::Episodic,
            Scope::Group,
            Some("g"),
            1.0,
            None,
            None,
            &test_embedding(10 + i),
            0.92,
        )
        .unwrap()
        .id;
        conn.execute(
            "UPDATE memories SET created_at = '2025-01-06T00:00:00+00:00' WHERE id = ?1",
            [&id],
        )
        .unwrap();
        ids.push(id);
    }
    let before = ops(&conn).len();
    let mut cfg = MaintenanceConfig::default();
    cfg.compaction_min_group_size = 3;
    cfg.compaction_age_days = 1;
    let res = compact_episodic(&mut conn, &FakeEmbedding, &cfg).unwrap();
    assert_eq!(res.summaries_created, 1);

    let new_ops: Vec<_> = ops(&conn).into_iter().skip(before).collect();
    assert_eq!(new_ops.len(), 1 + 3);
    let summary_id = match &new_ops[0].op {
        Op::MemoryPut { memory } => memory.id.clone(),
        other => panic!("expected summary put, got {other:?}"),
    };
    let mut superseded: Vec<String> = new_ops[1..]
        .iter()
        .map(|o| match &o.op {
            Op::MemorySupersede { old_id, new_id } => {
                assert_eq!(new_id, &summary_id);
                old_id.clone()
            }
            other => panic!("expected supersede, got {other:?}"),
        })
        .collect();
    superseded.sort();
    ids.sort();
    assert_eq!(superseded, ids);
}

#[test]
fn cleanup_is_local_but_tombstones_and_scrubs() {
    let mut conn = test_db();
    let id = store_memory(
        &mut conn,
        "synthetic stale needle-77",
        MemoryType::Episodic,
        Scope::Group,
        Some("g"),
        0.01,
        None,
        None,
        &test_embedding(3),
        0.92,
    )
    .unwrap()
    .id;
    conn.execute(
        "UPDATE memories SET created_at='2020-01-01T00:00:00+00:00', last_accessed='2020-01-01T00:00:00+00:00' WHERE id=?1",
        [&id],
    )
    .unwrap();
    let mut cfg = MaintenanceConfig::default();
    cfg.cleanup_confidence_floor = 0.1;
    cfg.cleanup_no_access_days = 30;
    let r = loci::memory::maintenance::cleanup_stale(&mut conn, &cfg, false).unwrap();
    assert_eq!(r.deleted, 1);
    assert!(
        ops(&conn)
            .iter()
            .all(|o| !matches!(o.op, Op::MemoryForget { .. }))
    );
    let leaked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM op_log WHERE body LIKE '%needle-77%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(leaked, 0);
    assert!(loci::sync::log::tombstone(&conn, &id).unwrap().unwrap().0);
}
