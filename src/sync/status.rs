//! Read-only replication status, surfaced through `memory_stats` and `loci stats`.

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

/// Last sync with one remote machine.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RemoteStatus {
    /// Remote `machine_id`.
    pub remote: String,
    /// ISO 8601 time of the last sync that read this remote.
    pub last_sync_at: Option<String>,
    /// `op_id` of the last op read from this remote.
    pub last_op_id: Option<String>,
}

/// An unresolved supersede conflict: `old_id` was superseded differently on two machines.
/// `winner_id` is what `superseded_by` points at; `other_id` is the competing successor.
/// Both successors remain active; consolidate them (e.g. supersede one with the other).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ForkStatus {
    pub old_id: String,
    pub winner_id: String,
    pub other_id: String,
}

/// Replication status of this database.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SyncStatus {
    /// This database's stable machine ID.
    pub machine_id: String,
    /// Last sync per remote machine.
    pub remotes: Vec<RemoteStatus>,
    /// Local ops not yet written to the sync file.
    pub pending_local_ops: u64,
    /// Remote ops waiting for a dependency (e.g. a relation whose endpoint is missing).
    pub deferred_ops: u64,
    /// Supersede conflicts nobody has consolidated yet.
    pub unresolved_forks: Vec<ForkStatus>,
}

/// Collect the replication status.
pub fn sync_status(conn: &Connection) -> Result<SyncStatus> {
    let machine_id = crate::db::migrations::get_machine_id(conn)?;
    let remotes = conn
        .prepare("SELECT remote, last_sync_at, last_op_id FROM sync_state ORDER BY remote")?
        .query_map([], |r| {
            Ok(RemoteStatus {
                remote: r.get(0)?,
                last_sync_at: r.get(1)?,
                last_op_id: r.get(2)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    let count =
        |sql: &str| -> Result<u64> { Ok(conn.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64) };
    let unresolved_forks = conn
        .prepare(
            "SELECT old_id, winner_id, other_id FROM supersede_forks WHERE resolved = 0 ORDER BY id",
        )?
        .query_map([], |r| {
            Ok(ForkStatus {
                old_id: r.get(0)?,
                winner_id: r.get(1)?,
                other_id: r.get(2)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(SyncStatus {
        machine_id,
        remotes,
        pending_local_ops: count("SELECT COUNT(*) FROM op_log WHERE exported = 0")?,
        deferred_ops: count("SELECT COUNT(*) FROM deferred_ops")?,
        unresolved_forks,
    })
}
