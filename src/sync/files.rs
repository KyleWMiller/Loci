//! Sync-directory file handling: own-file export, in-place scrub, remote reads.
//!
//! Each machine appends to its own `<machine_id>.jsonl` and only reads the others'.
//! Readers consume complete (newline-terminated) lines only, so a half-written last
//! line is ignored until it is finished.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::ops::{Op, OpEnvelope};

/// Characters allowed in a machine ID used as a file name.
pub fn valid_machine_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        && !id.starts_with('.')
}

/// Path of a machine's sync file.
pub fn machine_file(dir: &Path, machine_id: &str) -> PathBuf {
    dir.join(format!("{machine_id}.jsonl"))
}

/// Overwrite already-exported `memory.put` lines whose memory was hard-deleted.
///
/// The line is replaced by a `noop` op padded with spaces to the *same byte length*, so
/// every other machine's byte-offset cursor into this file stays valid.
/// Returns the number of lines scrubbed.
pub fn scrub_own_file(conn: &Connection, dir: &Path, machine_id: &str) -> Result<u64> {
    let rows: Vec<(i64, String, i64, i64)> = conn
        .prepare(
            "SELECT seq, body, file_offset, file_len FROM op_log \
             WHERE scrub_pending = 1 AND exported = 1 AND file_offset IS NOT NULL",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        return Ok(0);
    }

    let path = machine_file(dir, machine_id);
    let mut scrubbed = 0;
    if path.exists() {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        for (seq, body, offset, len) in &rows {
            let (offset, len) = (*offset as u64, *len as usize);
            if len < 2 || body.len() + 1 > len {
                bail!("cannot scrub op_log seq {seq}: replacement longer than original line");
            }
            // Only touch the line if it is still a complete line at that offset.
            let mut buf = vec![0u8; len];
            file.seek(SeekFrom::Start(offset))?;
            if file.read_exact(&mut buf).is_err() || buf.last() != Some(&b'\n') {
                tracing::warn!(
                    seq,
                    "sync file no longer matches recorded line; not scrubbed"
                );
                continue;
            }
            let mut line = body.clone().into_bytes();
            line.resize(len - 1, b' ');
            line.push(b'\n');
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&line)?;
            scrubbed += 1;
        }
        file.sync_all()?;
    }
    for (seq, ..) in &rows {
        conn.execute(
            "UPDATE op_log SET scrub_pending = 0 WHERE seq = ?1",
            params![seq],
        )?;
    }
    Ok(scrubbed)
}

/// Append not-yet-exported ops to our own file (fsync'd), then mark them exported.
/// Returns the number of ops written.
pub fn export_pending(conn: &mut Connection, dir: &Path, machine_id: &str) -> Result<u64> {
    let rows: Vec<(i64, String)> = conn
        .prepare("SELECT seq, body FROM op_log WHERE exported = 0 ORDER BY seq")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        return Ok(0);
    }

    let path = machine_file(dir, machine_id);
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut pos = file.metadata()?.len();

    // Terminate a line left unfinished by a crash so new ops start on their own line.
    if pos > 0 {
        let mut last = [0u8; 1];
        file.seek(SeekFrom::Start(pos - 1))?;
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            file.write_all(b"\n")?;
            pos += 1;
        }
    }

    let mut placed = Vec::with_capacity(rows.len());
    let mut buf = Vec::new();
    for (seq, body) in &rows {
        let start = pos + buf.len() as u64;
        buf.extend_from_slice(body.as_bytes());
        buf.push(b'\n');
        let len = pos + buf.len() as u64 - start;
        placed.push((*seq, start, len));
    }
    file.write_all(&buf)?;
    file.sync_all()?;

    let tx = conn.transaction()?;
    for (seq, offset, len) in &placed {
        tx.execute(
            "UPDATE op_log SET exported = 1, file_offset = ?1, file_len = ?2 WHERE seq = ?3",
            params![*offset as i64, *len as i64, seq],
        )?;
    }
    tx.commit()?;
    Ok(placed.len() as u64)
}

/// Remote machines' files in `dir` (everything `*.jsonl` except our own).
pub fn remote_files(dir: &Path, own_machine_id: &str) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem == own_machine_id || !valid_machine_id(stem) {
            continue;
        }
        out.push((stem.to_string(), path));
    }
    out.sort();
    Ok(out)
}

/// Lines read from a remote file plus the new cursor.
pub struct RemoteRead {
    pub ops: Vec<OpEnvelope>,
    /// Complete lines that did not parse as a supported op.
    pub malformed: u64,
    /// Byte offset just past the last complete line.
    pub new_offset: u64,
    /// The file was shorter than our cursor and was re-read from the start.
    pub reset: bool,
}

/// Read complete lines from `path` starting at `offset`.
pub fn read_remote(path: &Path, offset: u64) -> Result<RemoteRead> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let (start, reset) = if len < offset {
        (0, true)
    } else {
        (offset, false)
    };

    file.seek(SeekFrom::Start(start))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Consume complete lines only; a trailing partial line stays for the next sync.
    let end = data.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    let mut ops = Vec::new();
    let mut malformed = 0;
    for raw in data[..end].split(|b| *b == b'\n') {
        let text = String::from_utf8_lossy(raw);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        match serde_json::from_str::<OpEnvelope>(text) {
            Ok(env) if env.v == super::ops::OP_VERSION => ops.push(env),
            Ok(env) => {
                tracing::warn!(op_id = %env.op_id, v = env.v, "unsupported op version; skipped");
                malformed += 1;
            }
            Err(e) => {
                tracing::warn!(error = %e, "unparseable sync line; skipped");
                malformed += 1;
            }
        }
    }
    Ok(RemoteRead {
        ops,
        malformed,
        new_offset: start + end as u64,
        reset,
    })
}

/// A scrubbed placeholder line for `op_id` (used in tests and docs).
pub fn noop_line(env: &OpEnvelope) -> String {
    serde_json::to_string(&OpEnvelope {
        op: Op::Noop,
        ..env.clone()
    })
    .unwrap_or_default()
}
