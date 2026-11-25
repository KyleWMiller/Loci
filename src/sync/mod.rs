//! Replication between machines.
//!
//! IDs are preserved across machines and *operations* are replicated, one JSON line
//! per op, in a per-machine file `<sync_dir>/<machine_id>.jsonl`. A machine only ever
//! appends to its own file and only reads the others', so there are no write
//! conflicts and any transport that copies files works. See `docs/sync.md`.

pub mod apply;
pub mod files;
pub mod log;
pub mod ops;
pub mod status;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::fs::OpenOptions;
use std::path::Path;

use apply::{ApplyStats, Embedder, Outcome, apply_op, op_key};
use ops::OpEnvelope;

/// What a `loci sync` run did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SyncSummary {
    /// Local ops appended to our sync file.
    pub written: u64,
    /// Remote ops newly applied.
    pub applied: u64,
    /// Remote ops intentionally not applied (already known, deleted, invalid).
    pub skipped: u64,
    /// Ops still waiting for a dependency after this run.
    pub deferred: u64,
    /// New supersede forks recorded.
    pub forks: u64,
    /// New tombstones recorded.
    pub tombstones: u64,
    /// `memory.put` ops discarded because the ID is hard-deleted.
    pub discarded: u64,
    /// Complete lines that could not be parsed or are an unsupported version.
    pub malformed: u64,
    /// Lines in our own file overwritten because their memory was hard-deleted.
    pub scrubbed: u64,
    /// Remote files read.
    pub remotes: u64,
}

/// Override the stored machine ID (config `sync.machine_id`). Changing it is safe but
/// starts a new sync file; the old file is then read as a remote (idempotent).
pub fn set_machine_id_override(conn: &Connection, id: &str) -> Result<()> {
    if !files::valid_machine_id(id) {
        bail!("invalid machine_id {id:?}: use letters, digits, '-', '_' and '.'");
    }
    let current = crate::db::migrations::get_machine_id(conn)?;
    if current != id {
        tracing::warn!(from = %current, to = %id, "machine_id overridden by config");
        conn.execute(
            "UPDATE schema_meta SET value = ?1 WHERE key = 'machine_id'",
            params![id],
        )?;
    }
    Ok(())
}

/// Run one sync: apply remote ops, scrub our file, export our ops.
///
/// `embed` is only called for memories that are actually inserted.
pub fn run_sync(conn: &mut Connection, dir: &Path, embed: Embedder<'_>) -> Result<SyncSummary> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create sync dir {}", dir.display()))?;

    // Advisory lock so two syncs on this machine never interleave. Released on drop.
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(".loci-sync.lock"))?;
    lock.lock().context("failed to lock sync dir")?;

    let machine_id = crate::db::migrations::get_machine_id(conn)?;
    let mut summary = SyncSummary::default();

    apply_remotes(conn, dir, &machine_id, embed, &mut summary)?;
    summary.scrubbed = files::scrub_own_file(conn, dir, &machine_id)?;
    summary.written = files::export_pending(conn, dir, &machine_id)?;

    summary.deferred = conn.query_row("SELECT COUNT(*) FROM deferred_ops", [], |r| {
        r.get::<_, i64>(0)
    })? as u64;
    drop(lock);
    Ok(summary)
}

fn apply_remotes(
    conn: &mut Connection,
    dir: &Path,
    machine_id: &str,
    embed: Embedder<'_>,
    summary: &mut SyncSummary,
) -> Result<()> {
    // Read everything first; cursors only advance when the whole apply commits.
    let mut incoming: Vec<OpEnvelope> = Vec::new();
    let mut cursors: Vec<(String, u64, Option<String>)> = Vec::new();
    for (remote, path) in files::remote_files(dir, machine_id)? {
        let offset: u64 = conn
            .query_row(
                "SELECT byte_offset FROM sync_state WHERE remote = ?1",
                params![remote],
                |r| r.get::<_, i64>(0),
            )
            .map(|o| o as u64)
            .unwrap_or(0);
        let read = files::read_remote(&path, offset)?;
        if read.reset {
            tracing::warn!(remote = %remote, "remote file shrank; re-reading from the start");
        }
        summary.malformed += read.malformed;
        summary.remotes += 1;
        let last = read.ops.last().map(|o| o.op_id.clone());
        incoming.extend(read.ops.into_iter().filter(|o| o.origin != machine_id));
        cursors.push((remote, read.new_offset, last));
    }

    // Previously deferred ops are retried alongside the new ones.
    {
        let mut stmt = conn.prepare("SELECT body FROM deferred_ops")?;
        let rows: Vec<String> = stmt
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        for body in rows {
            if let Ok(env) = serde_json::from_str::<OpEnvelope>(&body) {
                incoming.push(env);
            }
        }
    }

    // Global (ts, op_id) order; drop duplicates of the same op.
    incoming.sort_by_key(|e| op_key(&e.ts, &e.op_id));
    incoming.dedup_by(|a, b| a.op_id == b.op_id);

    let tx = conn.transaction()?;
    let mut stats = ApplyStats::default();
    let mut pending: Vec<(OpEnvelope, &'static str)> = Vec::new();
    let mut queue = incoming;
    let now = chrono::Utc::now().to_rfc3339();

    // Repeat passes while deferred ops make progress (a put may unlock a relation).
    loop {
        let mut progressed = false;
        pending.clear();
        for env in queue {
            let already: bool = tx.query_row(
                "SELECT COUNT(*) > 0 FROM applied_ops WHERE op_id = ?1",
                params![env.op_id],
                |r| r.get(0),
            )?;
            if already {
                summary.skipped += 1;
                tx.execute(
                    "DELETE FROM deferred_ops WHERE op_id = ?1",
                    params![env.op_id],
                )?;
                continue;
            }
            let outcome = apply_op(&tx, &env, embed, &mut stats)?;
            match outcome {
                Outcome::Applied | Outcome::Skipped(_) => {
                    if outcome == Outcome::Applied {
                        summary.applied += 1;
                    } else {
                        summary.skipped += 1;
                    }
                    tx.execute(
                        "INSERT OR IGNORE INTO applied_ops (op_id, origin, applied_at) VALUES (?1, ?2, ?3)",
                        params![env.op_id, env.origin, now],
                    )?;
                    tx.execute(
                        "DELETE FROM deferred_ops WHERE op_id = ?1",
                        params![env.op_id],
                    )?;
                    progressed = true;
                }
                Outcome::Deferred(reason) => pending.push((env, reason)),
            }
        }
        if pending.is_empty() || !progressed {
            break;
        }
        queue = pending.drain(..).map(|(e, _)| e).collect();
    }

    for (env, reason) in &pending {
        tx.execute(
            "INSERT OR REPLACE INTO deferred_ops (op_id, ts, origin, body, reason, deferred_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                env.op_id,
                env.ts,
                env.origin,
                serde_json::to_string(env)?,
                reason,
                now
            ],
        )?;
    }

    for (remote, offset, last) in &cursors {
        tx.execute(
            "INSERT INTO sync_state (remote, byte_offset, last_op_id, last_sync_at) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(remote) DO UPDATE SET byte_offset = excluded.byte_offset, \
             last_op_id = COALESCE(excluded.last_op_id, sync_state.last_op_id), \
             last_sync_at = excluded.last_sync_at",
            params![remote, *offset as i64, last, now],
        )?;
    }
    tx.commit()?;

    summary.forks += stats.forks;
    summary.tombstones += stats.tombstones;
    summary.discarded += stats.discarded;
    Ok(())
}
