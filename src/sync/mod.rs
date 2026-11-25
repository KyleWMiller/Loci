//! Replication between machines: operation log, emission and (later) apply.
//!
//! See `docs/sync.md` for the model. In short: IDs are preserved across machines and
//! *operations* are replicated, one JSON line per op, in a per-machine file.

pub mod log;
pub mod ops;
