//! Idempotent application of ops received from other machines.
//!
//! Applying never emits ops (no echo) and never runs similarity dedup: replicated
//! memories keep their exact ID. See `docs/sync.md` for the conflict rules.

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

use super::log::{record_supersession, record_tombstone, scrub_memory_ops, tombstone};
use super::ops::{Op, OpEnvelope, PutMemory, PutRelation};
use crate::memory::relations::{RelationInsert, store_relation_with_id};
use crate::memory::store::{NewMemory, store_new_memory_tx, write_audit_log};
use crate::memory::types::{MemoryType, Scope};

/// Callback producing the embedding for a memory's content.
pub type Embedder<'a> = &'a mut dyn FnMut(&str) -> Result<Vec<f32>>;

/// What happened to one op.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Applied,
    /// Valid but intentionally not applied (already known, deleted, malformed...).
    Skipped(&'static str),
    /// Cannot be applied yet; retried on later syncs.
    Deferred(&'static str),
}

/// Counters accumulated while applying.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ApplyStats {
    pub applied: u64,
    pub skipped: u64,
    pub deferred: u64,
    /// New supersede forks recorded.
    pub forks: u64,
    /// New tombstones recorded.
    pub tombstones: u64,
    /// `memory.put` ops discarded because the ID is hard-deleted.
    pub discarded: u64,
}

/// Sort key for `(ts, op_id)` ordering. Unparseable timestamps sort first.
pub fn op_key(ts: &str, op_id: &str) -> (DateTime<Utc>, String) {
    let t = DateTime::parse_from_rfc3339(ts)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or(DateTime::<Utc>::MIN_UTC);
    (t, op_id.to_string())
}

fn rel_key(id: &str) -> String {
    format!("rel:{id}")
}

/// Apply one op inside `tx`.
pub fn apply_op(
    tx: &Transaction<'_>,
    env: &OpEnvelope,
    embed: Embedder<'_>,
    stats: &mut ApplyStats,
) -> Result<Outcome> {
    match &env.op {
        Op::Noop => Ok(Outcome::Applied),
        Op::MemoryPut { memory } => apply_put(tx, memory, embed, stats),
        Op::MemorySupersede { old_id, new_id } => apply_supersede(tx, env, old_id, new_id, stats),
        Op::MemoryForget { id, hard } => apply_forget(tx, env, id, *hard, stats),
        Op::RelationPut { relation } => apply_relation_put(tx, relation),
        Op::RelationDelete { id } => apply_relation_delete(tx, env, id, stats),
    }
}

fn memory_exists(conn: &Connection, id: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT COUNT(*) > 0 FROM memories WHERE id = ?1",
        params![id],
        |r| r.get(0),
    )?)
}

fn apply_put(
    tx: &Transaction<'_>,
    m: &PutMemory,
    embed: Embedder<'_>,
    stats: &mut ApplyStats,
) -> Result<Outcome> {
    let tomb = tombstone(tx, &m.id)?;
    if let Some((true, _)) = tomb {
        stats.discarded += 1;
        return Ok(Outcome::Skipped("hard-deleted"));
    }
    // Known ID: do nothing. Replicated fields never change after creation, so there
    // is nothing to merge (see docs/sync.md).
    if memory_exists(tx, &m.id)? {
        return Ok(Outcome::Skipped("already exists"));
    }

    let (Ok(memory_type), Ok(scope)) = (
        m.memory_type.parse::<MemoryType>(),
        m.scope.parse::<Scope>(),
    ) else {
        tracing::warn!(id = %m.id, "memory.put with unknown type/scope; skipped");
        return Ok(Outcome::Skipped("invalid type or scope"));
    };
    if !(0.0..=1.0).contains(&m.confidence) || m.content.is_empty() {
        return Ok(Outcome::Skipped("invalid memory fields"));
    }

    let embedding = embed(&m.content)?;
    let mut new = NewMemory::new(
        &m.content,
        memory_type,
        scope,
        m.source_group.as_deref(),
        m.confidence,
        m.metadata.as_ref(),
    );
    new.id = Some(&m.id);
    new.created_at = Some(&m.created_at);
    new.updated_at = Some(&m.updated_at);
    new.emit = false;
    store_new_memory_tx(tx, &new, &embedding, None)?;

    // A soft tombstone that arrived first: the memory exists, but forgotten.
    if tomb.is_some() {
        tx.execute(
            "UPDATE memories SET superseded_by = 'forgotten' WHERE id = ?1",
            params![m.id],
        )?;
    }
    Ok(Outcome::Applied)
}

fn apply_supersede(
    tx: &Transaction<'_>,
    env: &OpEnvelope,
    old_id: &str,
    new_id: &str,
    stats: &mut ApplyStats,
) -> Result<Outcome> {
    if let Some((true, _)) = tombstone(tx, old_id)? {
        return Ok(Outcome::Skipped("superseded memory hard-deleted"));
    }
    let current: Option<Option<String>> = tx
        .query_row(
            "SELECT superseded_by FROM memories WHERE id = ?1",
            params![old_id],
            |r| r.get(0),
        )
        .optional()?;

    match current {
        None => Ok(Outcome::Deferred("superseded memory not present yet")),
        Some(None) => {
            tx.execute(
                "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
                params![new_id, old_id],
            )?;
            record_supersession(tx, old_id, new_id, &env.ts, &env.op_id)?;
            write_audit_log(
                tx,
                "supersede",
                old_id,
                Some(&serde_json::json!({"superseded_by": new_id, "origin": env.origin})),
            )?;
            Ok(Outcome::Applied)
        }
        // A delete beats a supersession.
        Some(Some(cur)) if cur == "forgotten" => Ok(Outcome::Skipped("forgotten wins")),
        Some(Some(cur)) if cur == new_id => {
            tx.execute(
                "INSERT OR IGNORE INTO supersessions (old_id, new_id, ts, op_id) VALUES (?1, ?2, ?3, ?4)",
                params![old_id, new_id, env.ts, env.op_id],
            )?;
            Ok(Outcome::Skipped("already superseded by same memory"))
        }
        Some(Some(cur)) => {
            // Fork: two different supersessions of one memory. The earliest by
            // (ts, op_id) wins; both successors stay active; the fork is recorded.
            let existing: Option<(String, String)> = tx
                .query_row(
                    "SELECT ts, op_id FROM supersessions WHERE old_id = ?1 AND new_id = ?2",
                    params![old_id, cur],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            // A supersession with no record predates replication: it is the oldest.
            let cur_key = existing
                .map(|(ts, op)| op_key(&ts, &op))
                .unwrap_or((DateTime::<Utc>::MIN_UTC, String::new()));
            let inc_key = op_key(&env.ts, &env.op_id);

            let (winner, other) = if inc_key < cur_key {
                tx.execute(
                    "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
                    params![new_id, old_id],
                )?;
                record_supersession(tx, old_id, new_id, &env.ts, &env.op_id)?;
                (new_id.to_string(), cur)
            } else {
                (cur, new_id.to_string())
            };
            let inserted = tx.execute(
                "INSERT OR IGNORE INTO supersede_forks (old_id, winner_id, other_id, detected_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![old_id, winner, other, chrono::Utc::now().to_rfc3339()],
            )?;
            stats.forks += inserted as u64;
            Ok(Outcome::Applied)
        }
    }
}

/// Hard-delete a memory's rows (memories, FTS, vec; relations cascade). Returns
/// whether it existed.
fn hard_delete_rows(tx: &Transaction<'_>, id: &str) -> Result<bool> {
    let row: Option<(i64, String, String)> = tx
        .query_row(
            "SELECT rowid, content, type FROM memories WHERE id = ?1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((rowid, content, memory_type)) = row else {
        return Ok(false);
    };
    tx.execute(
        "INSERT INTO memories_fts(memories_fts, rowid, content, id, type) VALUES('delete', ?1, ?2, ?3, ?4)",
        params![rowid, content, id, memory_type],
    )?;
    tx.execute("DELETE FROM memories_vec WHERE id = ?1", params![id])?;
    // The reason is deliberately not replicated, so the audit row has none.
    write_audit_log(
        tx,
        "delete",
        id,
        Some(&serde_json::json!({"hard_delete": true, "origin": "sync"})),
    )?;
    tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
    Ok(true)
}

fn apply_forget(
    tx: &Transaction<'_>,
    env: &OpEnvelope,
    id: &str,
    hard: bool,
    stats: &mut ApplyStats,
) -> Result<Outcome> {
    let before = tombstone(tx, id)?;
    record_tombstone(tx, id, hard, &env.ts, Some(&env.op_id), Some(&env.origin))?;
    let after = tombstone(tx, id)?;
    if before.is_none() || before.map(|b| b.0) != after.map(|a| a.0) {
        stats.tombstones += 1;
    }

    if hard {
        hard_delete_rows(tx, id)?;
        // Content must not survive in our op log, deferred ops or sync file.
        scrub_memory_ops(tx, id)?;
        return Ok(Outcome::Applied);
    }

    // Soft forget: never downgrade a hard-deleted memory (it no longer exists anyway).
    if memory_exists(tx, id)? {
        tx.execute(
            "UPDATE memories SET superseded_by = 'forgotten' WHERE id = ?1",
            params![id],
        )?;
        write_audit_log(
            tx,
            "delete",
            id,
            Some(&serde_json::json!({"hard_delete": false, "origin": "sync"})),
        )?;
    }
    Ok(Outcome::Applied)
}

fn apply_relation_put(tx: &Transaction<'_>, r: &PutRelation) -> Result<Outcome> {
    if tombstone(tx, &rel_key(&r.id))?.is_some() {
        return Ok(Outcome::Skipped("relation deleted"));
    }
    for endpoint in [&r.subject_id, &r.object_id] {
        if let Some((true, _)) = tombstone(tx, endpoint)? {
            return Ok(Outcome::Skipped("endpoint hard-deleted"));
        }
        if !memory_exists(tx, endpoint)? {
            return Ok(Outcome::Deferred("relation endpoint not present yet"));
        }
    }
    match store_relation_with_id(
        tx,
        &r.id,
        &r.subject_id,
        &r.predicate,
        &r.object_id,
        &r.created_at,
    ) {
        Ok(RelationInsert::Inserted) => Ok(Outcome::Applied),
        Ok(RelationInsert::Existing) => Ok(Outcome::Skipped("already exists")),
        Err(e) => {
            tracing::warn!(relation = %r.id, error = %e, "invalid relation.put; skipped");
            Ok(Outcome::Skipped("invalid relation"))
        }
    }
}

fn apply_relation_delete(
    tx: &Transaction<'_>,
    env: &OpEnvelope,
    id: &str,
    stats: &mut ApplyStats,
) -> Result<Outcome> {
    let key = rel_key(id);
    if tombstone(tx, &key)?.is_none() {
        stats.tombstones += 1;
    }
    record_tombstone(tx, &key, true, &env.ts, Some(&env.op_id), Some(&env.origin))?;
    tx.execute("DELETE FROM entity_relations WHERE id = ?1", params![id])?;
    Ok(Outcome::Applied)
}
