//! Step 5: `loci sync` between two (or three) databases through a shared directory.
//! All content is synthetic.

mod helpers;

use helpers::{test_db, test_embedding};
use loci::memory::forget::forget_memory;
use loci::memory::relations::store_relation;
use loci::memory::store::store_memory;
use loci::memory::types::{MemoryType, Scope};
use loci::sync::{SyncSummary, run_sync};
use rusqlite::Connection;
use std::path::Path;

fn embed(text: &str) -> anyhow::Result<Vec<f32>> {
    let seed = text
        .bytes()
        .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
    Ok(test_embedding((seed % 251) as u8))
}

fn sync(conn: &mut Connection, dir: &Path) -> SyncSummary {
    run_sync(conn, dir, &mut embed).unwrap()
}

/// A syncs, B syncs, A syncs, B syncs: both directions, twice.
fn full_cycle(a: &mut Connection, b: &mut Connection, dir: &Path) {
    for _ in 0..2 {
        sync(a, dir);
        sync(b, dir);
    }
}

fn put(conn: &mut Connection, content: &str, t: MemoryType, seed: u8, sup: Option<&str>) -> String {
    store_memory(
        conn,
        content,
        t,
        Scope::Global,
        Some("synthetic"),
        0.9,
        Some(&serde_json::json!({"s": "synthetic"})),
        sup,
        &test_embedding(seed),
        0.92,
    )
    .unwrap()
    .id
}

type State = (
    Vec<(String, String, String, Option<String>, String)>,
    Vec<(String, String, String, String)>,
);

/// Logical state: IDs, content, superseded_by, relation triples (no access stats).
fn state(conn: &Connection) -> State {
    let mems = conn
        .prepare("SELECT id, type, content, superseded_by, COALESCE(metadata,'') FROM memories ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let rels = conn
        .prepare("SELECT id, subject_id, predicate, object_id FROM entity_relations ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    (mems, rels)
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

fn machine(conn: &Connection) -> String {
    loci::db::migrations::get_machine_id(conn).unwrap()
}

fn dir_text(dir: &Path) -> String {
    let mut all = String::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
            all.push_str(&String::from_utf8_lossy(&std::fs::read(&p).unwrap()));
        }
    }
    all
}

#[test]
fn concurrent_writes_converge_and_resync_is_a_noop() {
    let dir = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    assert_ne!(machine(&a), machine(&b));

    // Both machines write while "offline".
    let ea = put(
        &mut a,
        "synthetic entity alpha",
        MemoryType::Entity,
        1,
        None,
    );
    let eb = put(&mut b, "synthetic entity beta", MemoryType::Entity, 2, None);
    let old = put(&mut a, "synthetic fact one", MemoryType::Semantic, 3, None);
    put(
        &mut a,
        "synthetic fact one revised",
        MemoryType::Semantic,
        4,
        Some(&old),
    );
    put(
        &mut b,
        "synthetic procedure",
        MemoryType::Procedural,
        5,
        None,
    );
    let gone = put(
        &mut b,
        "synthetic soft forgotten",
        MemoryType::Semantic,
        6,
        None,
    );
    forget_memory(&mut b, &gone, None, false).unwrap();

    full_cycle(&mut a, &mut b, dir.path());
    // A relation across machines, created after both have both entities.
    store_relation(&a, &ea, "knows", &eb).unwrap();
    full_cycle(&mut a, &mut b, dir.path());

    assert_eq!(state(&a), state(&b));
    assert_eq!(count(&a, "SELECT COUNT(*) FROM memories"), 6);
    assert_eq!(count(&b, "SELECT COUNT(*) FROM memories_vec"), 6);
    assert_eq!(
        count(
            &b,
            "SELECT COUNT(*) FROM entity_relations WHERE predicate='knows'"
        ),
        1
    );
    // Supersession and soft-forget travelled.
    assert_eq!(
        count(
            &b,
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
        ),
        2
    );

    // Third sync with no new ops writes and applies nothing.
    let before = (state(&a), count(&a, "SELECT COUNT(*) FROM memory_log"));
    let s = sync(&mut a, dir.path());
    assert_eq!((s.written, s.applied, s.deferred, s.forks), (0, 0, 0, 0));
    let s = sync(&mut b, dir.path());
    assert_eq!((s.written, s.applied, s.deferred, s.forks), (0, 0, 0, 0));
    assert_eq!(
        before,
        (state(&a), count(&a, "SELECT COUNT(*) FROM memory_log"))
    );
}

#[test]
fn applied_ops_are_not_echoed_back() {
    let dir = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    put(&mut a, "synthetic only on a", MemoryType::Semantic, 1, None);
    full_cycle(&mut a, &mut b, dir.path());
    assert_eq!(count(&b, "SELECT COUNT(*) FROM memories"), 1);
    assert_eq!(count(&b, "SELECT COUNT(*) FROM op_log"), 0);
    let bfile = dir.path().join(format!("{}.jsonl", machine(&b)));
    assert!(!bfile.exists() || std::fs::read(&bfile).unwrap().is_empty());
}

#[test]
fn losing_the_cursor_is_harmless() {
    let dir = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    let e = put(&mut a, "synthetic entity", MemoryType::Entity, 1, None);
    let e2 = put(&mut a, "synthetic entity two", MemoryType::Entity, 2, None);
    store_relation(&a, &e, "rel", &e2).unwrap();
    full_cycle(&mut a, &mut b, dir.path());
    let before = state(&b);
    let log_before = count(&b, "SELECT COUNT(*) FROM memory_log");
    b.execute("DELETE FROM sync_state", []).unwrap();
    let s = sync(&mut b, dir.path());
    assert_eq!(s.applied, 0);
    assert_eq!(before, state(&b));
    assert_eq!(log_before, count(&b, "SELECT COUNT(*) FROM memory_log"));
    // Same with the applied-op memory gone too: rules alone keep it idempotent.
    b.execute("DELETE FROM sync_state", []).unwrap();
    b.execute("DELETE FROM applied_ops", []).unwrap();
    sync(&mut b, dir.path());
    assert_eq!(before, state(&b));
}

#[test]
fn conflicting_supersessions_fork_with_same_winner_on_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    let x = put(&mut a, "synthetic base fact", MemoryType::Semantic, 1, None);
    full_cycle(&mut a, &mut b, dir.path());

    // Offline: each side supersedes X differently.
    let y = put(
        &mut a,
        "synthetic successor on a",
        MemoryType::Semantic,
        2,
        Some(&x),
    );
    let z = put(
        &mut b,
        "synthetic successor on b",
        MemoryType::Semantic,
        3,
        Some(&x),
    );
    full_cycle(&mut a, &mut b, dir.path());

    // Expected winner: earliest supersede op by (ts, op_id).
    let key = |c: &Connection| -> (String, String, String) {
        c.query_row(
            "SELECT ts, op_id, json_extract(body, '$.new_id') FROM op_log WHERE kind='memory.supersede'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap()
    };
    let (ka, kb) = (key(&a), key(&b));
    let expected = if (ka.0.as_str(), ka.1.as_str()) < (kb.0.as_str(), kb.1.as_str()) {
        ka.2.clone()
    } else {
        kb.2.clone()
    };

    for c in [&a, &b] {
        let sb: String = c
            .query_row(
                "SELECT superseded_by FROM memories WHERE id = ?1",
                [&x],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(sb, expected);
        assert_eq!(count(c, "SELECT COUNT(*) FROM supersede_forks"), 1);
        let (w, o): (String, String) = c
            .query_row("SELECT winner_id, other_id FROM supersede_forks", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(w, expected);
        assert_eq!(w == y, o == z);
        // Both successors stay active.
        for id in [&y, &z] {
            let n: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = ?1 AND superseded_by IS NULL",
                    [id],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1);
        }
    }
    assert_eq!(state(&a), state(&b));
}

#[test]
fn hard_delete_removes_content_everywhere_including_sync_files() {
    let dir = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    // B authors the memory, so B's own sync file holds the content.
    let id = put(
        &mut b,
        "synthetic secret needle-31337 locator",
        MemoryType::Semantic,
        1,
        None,
    );
    put(
        &mut b,
        "synthetic unrelated keeper",
        MemoryType::Semantic,
        2,
        None,
    );
    full_cycle(&mut a, &mut b, dir.path());
    assert_eq!(count(&a, "SELECT COUNT(*) FROM memories"), 2);
    assert!(dir_text(dir.path()).contains("needle-31337"));
    let bfile = dir.path().join(format!("{}.jsonl", machine(&b)));
    let len_before = std::fs::metadata(&bfile).unwrap().len();

    // A hard-deletes after B has the memory.
    forget_memory(&mut a, &id, Some("synthetic reason"), true).unwrap();
    full_cycle(&mut a, &mut b, dir.path());

    for (name, c) in [("a", &a), ("b", &b)] {
        assert_eq!(count(c, "SELECT COUNT(*) FROM memories"), 1, "{name}");
        assert_eq!(count(c, "SELECT COUNT(*) FROM memories_vec"), 1, "{name}");
        assert_eq!(
            count(
                c,
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'needle'"
            ),
            0,
            "{name} fts"
        );
        for (t, col) in [
            ("memory_log", "details"),
            ("op_log", "body"),
            ("deferred_ops", "body"),
        ] {
            assert_eq!(
                count(
                    c,
                    &format!("SELECT COUNT(*) FROM {t} WHERE {col} LIKE '%needle-31337%'")
                ),
                0,
                "{name} {t}"
            );
        }
        assert_eq!(
            count(
                c,
                &format!("SELECT hard FROM tombstones WHERE memory_id = '{id}'")
            ),
            1
        );
    }
    // Content is gone from every file in the sync dir, and B's file kept its length
    // (in-place scrub), so cursors stay valid.
    assert!(!dir_text(dir.path()).contains("needle-31337"));
    assert_eq!(std::fs::metadata(&bfile).unwrap().len(), len_before);

    // A cursor that was already past the scrubbed line keeps working afterwards.
    let later = put(
        &mut b,
        "synthetic written after the scrub",
        MemoryType::Semantic,
        9,
        None,
    );
    full_cycle(&mut a, &mut b, dir.path());
    assert_eq!(
        count(
            &a,
            &format!("SELECT COUNT(*) FROM memories WHERE id = '{later}'")
        ),
        1
    );
    assert_eq!(state(&a), state(&b));
}

#[test]
fn put_arriving_after_a_hard_tombstone_stays_deleted() {
    let src = tempfile::tempdir().unwrap();
    let dir_c = tempfile::tempdir().unwrap();
    let (mut a, mut b, mut c) = (test_db(), test_db(), test_db());
    let id = put(&mut a, "synthetic doomed", MemoryType::Semantic, 1, None);
    full_cycle(&mut a, &mut b, src.path());
    forget_memory(&mut b, &id, None, true).unwrap();
    sync(&mut b, src.path());

    let fa = format!("{}.jsonl", machine(&a));
    let fb = format!("{}.jsonl", machine(&b));
    // C sees B's forget first, then A's put.
    std::fs::copy(src.path().join(&fb), dir_c.path().join(&fb)).unwrap();
    sync(&mut c, dir_c.path());
    assert_eq!(count(&c, "SELECT COUNT(*) FROM tombstones"), 1);

    // A's file as B left it: A's put line is still there only if A has not scrubbed.
    std::fs::copy(src.path().join(&fa), dir_c.path().join(&fa)).unwrap();
    let s = sync(&mut c, dir_c.path());
    assert_eq!(count(&c, "SELECT COUNT(*) FROM memories"), 0);
    assert_eq!(count(&c, "SELECT COUNT(*) FROM memories_vec"), 0);
    assert_eq!(s.discarded, 1, "{s:?}");
}

#[test]
fn soft_forget_arriving_before_put_ends_up_forgotten() {
    let src = tempfile::tempdir().unwrap();
    let dir_c = tempfile::tempdir().unwrap();
    let (mut a, mut b, mut c) = (test_db(), test_db(), test_db());
    let id = put(
        &mut a,
        "synthetic soft doomed",
        MemoryType::Semantic,
        1,
        None,
    );
    full_cycle(&mut a, &mut b, src.path());
    forget_memory(&mut b, &id, None, false).unwrap();
    sync(&mut b, src.path());

    let fa = format!("{}.jsonl", machine(&a));
    let fb = format!("{}.jsonl", machine(&b));
    std::fs::copy(src.path().join(&fb), dir_c.path().join(&fb)).unwrap();
    sync(&mut c, dir_c.path());
    std::fs::copy(src.path().join(&fa), dir_c.path().join(&fa)).unwrap();
    sync(&mut c, dir_c.path());
    let sb: String = c
        .query_row(
            "SELECT superseded_by FROM memories WHERE id = ?1",
            [&id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(sb, "forgotten");
}

#[test]
fn relation_before_its_endpoint_is_deferred_then_applied() {
    let src = tempfile::tempdir().unwrap();
    let dir_c = tempfile::tempdir().unwrap();
    let (mut a, mut b, mut c) = (test_db(), test_db(), test_db());
    let e2 = put(&mut b, "synthetic entity on b", MemoryType::Entity, 2, None);
    sync(&mut b, src.path());
    sync(&mut a, src.path()); // A now has e2
    let e1 = put(&mut a, "synthetic entity on a", MemoryType::Entity, 1, None);
    store_relation(&a, &e1, "links", &e2).unwrap();
    sync(&mut a, src.path());

    let fa = format!("{}.jsonl", machine(&a));
    let fb = format!("{}.jsonl", machine(&b));
    // C only has A's file: e1 and the relation arrive, but e2 (in B's file) does not.
    std::fs::copy(src.path().join(&fa), dir_c.path().join(&fa)).unwrap();
    let s = sync(&mut c, dir_c.path());
    assert_eq!(s.deferred, 1);
    assert_eq!(count(&c, "SELECT COUNT(*) FROM entity_relations"), 0);

    // Re-syncing without new data keeps it deferred, not lost or duplicated.
    let s = sync(&mut c, dir_c.path());
    assert_eq!(s.deferred, 1);

    std::fs::copy(src.path().join(&fb), dir_c.path().join(&fb)).unwrap();
    let s = sync(&mut c, dir_c.path());
    assert_eq!(s.deferred, 0);
    assert_eq!(count(&c, "SELECT COUNT(*) FROM entity_relations"), 1);
    assert_eq!(count(&c, "SELECT COUNT(*) FROM deferred_ops"), 0);
    assert_eq!(state(&c), state(&a));
}

#[test]
fn truncated_last_line_is_ignored_then_picked_up() {
    let src = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let (mut a, mut b) = (test_db(), test_db());
    put(&mut a, "synthetic first", MemoryType::Semantic, 1, None);
    put(&mut a, "synthetic second", MemoryType::Semantic, 2, None);
    sync(&mut a, src.path());

    let fa = format!("{}.jsonl", machine(&a));
    let full = std::fs::read(src.path().join(&fa)).unwrap();
    let cut = full.len() - 25; // mid-way through the last line, no trailing newline
    std::fs::write(dir_b.path().join(&fa), &full[..cut]).unwrap();
    let s = sync(&mut b, dir_b.path());
    assert_eq!((s.applied, s.malformed), (1, 0));
    assert_eq!(count(&b, "SELECT COUNT(*) FROM memories"), 1);

    std::fs::write(dir_b.path().join(&fa), &full).unwrap();
    let s = sync(&mut b, dir_b.path());
    assert_eq!((s.applied, s.malformed), (1, 0));
    assert_eq!(count(&b, "SELECT COUNT(*) FROM memories"), 2);
    let off: i64 = b
        .query_row("SELECT byte_offset FROM sync_state", [], |r| r.get(0))
        .unwrap();
    assert_eq!(off as usize, full.len());
}

#[test]
fn garbage_lines_are_counted_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let mut b = test_db();
    std::fs::write(
        dir.path().join("peer.jsonl"),
        "not json\n{\"v\":9,\"op_id\":\"x\"}\n",
    )
    .unwrap();
    let s = sync(&mut b, dir.path());
    assert_eq!(s.malformed, 2);
}

#[test]
fn machine_id_override_validates_and_renames_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = test_db();
    assert!(loci::sync::set_machine_id_override(&a, "bad/id").is_err());
    loci::sync::set_machine_id_override(&a, "laptop").unwrap();
    put(&mut a, "synthetic named", MemoryType::Semantic, 1, None);
    sync(&mut a, dir.path());
    assert!(dir.path().join("laptop.jsonl").exists());
}
