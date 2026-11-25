//! Local op log: emission inside the caller's transaction, tombstones, and the
//! hard-delete scrub that keeps deleted content out of the log and sync files.

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};

use super::ops::{OP_VERSION, Op, OpEnvelope, PutMemory};

/// Append an op to `op_log` using the caller's transaction/connection.
///
/// Callers pass the same transaction as the mutation so the state change and its op
/// commit (or roll back) together.
pub fn emit(conn: &Connection, op: Op) -> Result<()> {
    let origin = crate::db::migrations::get_machine_id(conn)?;
    let env = OpEnvelope {
        v: OP_VERSION,
        op_id: uuid::Uuid::now_v7().to_string(),
        ts: chrono::Utc::now().to_rfc3339(),
        origin,
        op,
    };
    if let Op::MemorySupersede { old_id, new_id } = &env.op {
        record_supersession(conn, old_id, new_id, &env.ts, &env.op_id)?;
    }
    insert_op_log(conn, &env)
}

/// Remember which op (by `ts`, `op_id`) set `old_id`'s `superseded_by`.
pub fn record_supersession(
    conn: &Connection,
    old_id: &str,
    new_id: &str,
    ts: &str,
    op_id: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO supersessions (old_id, new_id, ts, op_id) VALUES (?1, ?2, ?3, ?4)",
        params![old_id, new_id, ts, op_id],
    )?;
    Ok(())
}

fn insert_op_log(conn: &Connection, env: &OpEnvelope) -> Result<()> {
    conn.execute(
        "INSERT INTO op_log (op_id, ts, origin, kind, memory_id, body) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            env.op_id,
            env.ts,
            env.origin,
            env.op.kind(),
            env.op.memory_id(),
            serde_json::to_string(env)?,
        ],
    )?;
    Ok(())
}

/// Emit a `memory.put` for the stored row `id` (reads the row back so the op carries
/// exactly what was persisted).
pub fn emit_put_for(conn: &Connection, id: &str) -> Result<()> {
    let memory: PutMemory = conn.query_row(
        "SELECT id, type, content, source_group, scope, confidence, created_at, updated_at, metadata \
         FROM memories WHERE id = ?1",
        params![id],
        |row| {
            let metadata: Option<String> = row.get(8)?;
            Ok(PutMemory {
                id: row.get(0)?,
                memory_type: row.get(1)?,
                content: row.get(2)?,
                source_group: row.get(3)?,
                scope: row.get(4)?,
                confidence: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
                metadata: metadata.and_then(|s| serde_json::from_str(&s).ok()),
            })
        },
    )?;
    emit(conn, Op::MemoryPut { memory })
}

/// Record (or upgrade) a tombstone. A hard tombstone is never downgraded.
pub fn record_tombstone(
    conn: &Connection,
    memory_id: &str,
    hard: bool,
    ts: &str,
    op_id: Option<&str>,
    origin: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO tombstones (memory_id, hard, ts, op_id, origin) VALUES (?1, ?2, ?3, ?4, ?5) \
         ON CONFLICT(memory_id) DO UPDATE SET hard = MAX(hard, excluded.hard) \
         WHERE excluded.hard > tombstones.hard",
        params![memory_id, hard as i64, ts, op_id, origin],
    )?;
    Ok(())
}

/// The tombstone for `memory_id`, as `(hard, ts)`.
pub fn tombstone(conn: &Connection, memory_id: &str) -> Result<Option<(bool, String)>> {
    Ok(conn
        .query_row(
            "SELECT hard, ts FROM tombstones WHERE memory_id = ?1",
            params![memory_id],
            |row| Ok((row.get::<_, i64>(0)? != 0, row.get::<_, String>(1)?)),
        )
        .optional()?)
}

/// Remove every trace of a hard-deleted memory's *content* from the op log.
///
/// - Not-yet-exported `memory.put` rows are deleted outright.
/// - Already-exported ones become `noop` rows with `scrub_pending = 1`; the next
///   `loci sync` overwrites that line in the sync file in place (same byte length, so
///   other machines' read cursors stay valid).
/// - Deferred remote ops carrying the content are dropped.
pub fn scrub_memory_ops(conn: &Connection, memory_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM op_log WHERE memory_id = ?1 AND kind = 'memory.put' AND exported = 0",
        params![memory_id],
    )?;

    let mut stmt = conn.prepare(
        "SELECT seq, op_id, ts, origin FROM op_log \
         WHERE memory_id = ?1 AND kind = 'memory.put' AND exported = 1",
    )?;
    let rows: Vec<(i64, String, String, String)> = stmt
        .query_map(params![memory_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })?
        .collect::<Result<_, _>>()?;
    drop(stmt);
    for (seq, op_id, ts, origin) in rows {
        let noop = OpEnvelope {
            v: OP_VERSION,
            op_id,
            ts,
            origin,
            op: Op::Noop,
        };
        conn.execute(
            "UPDATE op_log SET kind = 'noop', memory_id = NULL, body = ?1, scrub_pending = 1 \
             WHERE seq = ?2",
            params![serde_json::to_string(&noop)?, seq],
        )?;
    }

    // Deferred remote ops that still hold this memory's content.
    conn.execute(
        "DELETE FROM deferred_ops WHERE json_extract(body, '$.memory.id') = ?1",
        params![memory_id],
    )?;
    Ok(())
}

/// Scrub every content-bearing op (used by `loci reset`).
pub fn scrub_all_ops(conn: &Connection) -> Result<()> {
    conn.execute(
        "DELETE FROM op_log WHERE kind = 'memory.put' AND exported = 0",
        [],
    )?;
    let ids: Vec<String> = conn
        .prepare(
            "SELECT DISTINCT memory_id FROM op_log WHERE kind = 'memory.put' AND exported = 1",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    for id in ids {
        scrub_memory_ops(conn, &id)?;
    }
    conn.execute("DELETE FROM deferred_ops", [])?;
    Ok(())
}
