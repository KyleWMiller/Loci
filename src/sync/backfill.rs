//! One-time seeding of the op log from data that predates replication.
//!
//! Memories, supersessions, soft-forgets and relations created by 0.2.x (or imported
//! before the first sync) have no ops. The first `loci sync` emits them once, guarded by
//! the `sync_backfill` marker in `schema_meta`, so they reach the other machines.

use anyhow::Result;
use rusqlite::{Connection, params};

use super::log::{emit, emit_put_for, record_tombstone};
use super::ops::{Op, PutRelation};

const MARKER: &str = "sync_backfill";

/// Emit ops for existing rows that have none. Returns the number of ops emitted;
/// does nothing if the backfill already ran.
pub fn backfill_existing(conn: &Connection) -> Result<u64> {
    let done: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM schema_meta WHERE key = ?1",
        params![MARKER],
        |r| r.get(0),
    )?;
    if done {
        return Ok(0);
    }
    let mut emitted = 0;

    // Memories with no put in the log.
    let ids: Vec<String> = conn
        .prepare(
            "SELECT id FROM memories m WHERE NOT EXISTS \
             (SELECT 1 FROM op_log o WHERE o.kind = 'memory.put' AND o.memory_id = m.id) \
             ORDER BY created_at, id",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    for id in &ids {
        emit_put_for(conn, id)?;
        emitted += 1;
    }

    // Supersession / soft-forget state with no record.
    let chains: Vec<(String, String)> = conn
        .prepare(
            "SELECT id, superseded_by FROM memories m WHERE superseded_by IS NOT NULL \
             AND NOT EXISTS (SELECT 1 FROM supersessions s WHERE s.old_id = m.id) \
             AND NOT EXISTS (SELECT 1 FROM tombstones t WHERE t.memory_id = m.id) \
             ORDER BY id",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    for (id, target) in chains {
        if target == "forgotten" {
            record_tombstone(
                conn,
                &id,
                false,
                &chrono::Utc::now().to_rfc3339(),
                None,
                None,
            )?;
            emit(conn, Op::MemoryForget { id, hard: false })?;
        } else {
            emit(
                conn,
                Op::MemorySupersede {
                    old_id: id,
                    new_id: target,
                },
            )?;
        }
        emitted += 1;
    }

    // Relations with no put in the log.
    let rels: Vec<PutRelation> = conn
        .prepare(
            "SELECT id, subject_id, predicate, object_id, created_at FROM entity_relations r \
             WHERE NOT EXISTS (SELECT 1 FROM op_log o WHERE o.kind = 'relation.put' \
                               AND json_extract(o.body, '$.relation.id') = r.id) \
             ORDER BY created_at, id",
        )?
        .query_map([], |r| {
            Ok(PutRelation {
                id: r.get(0)?,
                subject_id: r.get(1)?,
                predicate: r.get(2)?,
                object_id: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    for relation in rels {
        emit(conn, Op::RelationPut { relation })?;
        emitted += 1;
    }

    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES (?1, ?2)",
        params![MARKER, chrono::Utc::now().to_rfc3339()],
    )?;
    Ok(emitted)
}
