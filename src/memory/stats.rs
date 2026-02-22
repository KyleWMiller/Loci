use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

/// Response from memory_stats.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_memories: u64,
    pub active_memories: u64,
    pub superseded_memories: u64,
    pub by_type: HashMap<String, u64>,
    pub by_scope: HashMap<String, u64>,
    pub entity_relations: u64,
    pub db_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oldest_memory: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_memory: Option<String>,
}

/// Compute memory store statistics.
///
/// If `group` is provided, counts are filtered to that group (plus global-scope memories).
/// `db_path` is used for file size calculation; pass None for in-memory databases.
pub fn memory_stats(
    conn: &Connection,
    group: Option<&str>,
    db_path: Option<&Path>,
) -> Result<StatsResponse> {
    let (total, active, superseded) = count_memories(conn, group)?;
    let by_type = count_by_type(conn, group)?;
    let by_scope = count_by_scope(conn, group)?;
    let entity_relations = count_relations(conn)?;
    let (oldest, newest) = memory_time_range(conn, group)?;

    let db_size_bytes = db_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(StatsResponse {
        total_memories: total,
        active_memories: active,
        superseded_memories: superseded,
        by_type,
        by_scope,
        entity_relations,
        db_size_bytes,
        oldest_memory: oldest,
        newest_memory: newest,
    })
}

/// Total, active, and superseded counts.
fn count_memories(conn: &Connection, group: Option<&str>) -> Result<(u64, u64, u64)> {
    let (where_clause, param) = group_filter(group);

    let total: i64 = if let Some(ref g) = param {
        conn.query_row(
            &format!("SELECT COUNT(*) FROM memories {where_clause}"),
            params![g],
            |row| row.get(0),
        )?
    } else {
        conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?
    };

    let active: i64 = if let Some(ref g) = param {
        conn.query_row(
            &format!("SELECT COUNT(*) FROM memories {where_clause} AND superseded_by IS NULL"),
            params![g],
            |row| row.get(0),
        )?
    } else {
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL",
            [],
            |row| row.get(0),
        )?
    };

    let superseded = total - active;
    Ok((total as u64, active as u64, superseded as u64))
}

/// Count by memory type.
fn count_by_type(conn: &Connection, group: Option<&str>) -> Result<HashMap<String, u64>> {
    let (where_clause, param) = group_filter(group);
    let sql = format!("SELECT type, COUNT(*) FROM memories {where_clause} GROUP BY type");

    let mut map = HashMap::new();
    for t in &["episodic", "semantic", "procedural", "entity"] {
        map.insert(t.to_string(), 0);
    }

    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<(String, i64)> = if let Some(ref g) = param {
        stmt.query_map(params![g], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?
    };

    for (t, count) in rows {
        map.insert(t, count as u64);
    }
    Ok(map)
}

/// Count by scope.
fn count_by_scope(conn: &Connection, group: Option<&str>) -> Result<HashMap<String, u64>> {
    let (where_clause, param) = group_filter(group);
    let sql = format!("SELECT scope, COUNT(*) FROM memories {where_clause} GROUP BY scope");

    let mut map = HashMap::new();
    for s in &["global", "group"] {
        map.insert(s.to_string(), 0);
    }

    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<(String, i64)> = if let Some(ref g) = param {
        stmt.query_map(params![g], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?
    };

    for (s, count) in rows {
        map.insert(s, count as u64);
    }
    Ok(map)
}

/// Count total entity relations.
fn count_relations(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM entity_relations",
        [],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

/// Oldest and newest memory timestamps.
fn memory_time_range(
    conn: &Connection,
    group: Option<&str>,
) -> Result<(Option<String>, Option<String>)> {
    let (where_clause, param) = group_filter(group);
    let sql = format!(
        "SELECT MIN(created_at), MAX(created_at) FROM memories {where_clause}"
    );

    let (oldest, newest): (Option<String>, Option<String>) = if let Some(ref g) = param {
        conn.query_row(&sql, params![g], |row| Ok((row.get(0)?, row.get(1)?)))?
    } else {
        conn.query_row(&sql, [], |row| Ok((row.get(0)?, row.get(1)?)))?
    };

    Ok((oldest, newest))
}

/// Build a WHERE clause for optional group filtering.
///
/// When a group is provided, includes memories from that group plus global-scope memories.
fn group_filter(group: Option<&str>) -> (String, Option<String>) {
    match group {
        Some(g) => (
            "WHERE (source_group = ?1 OR scope = 'global')".to_string(),
            Some(g.to_string()),
        ),
        None => (String::new(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::memory::store;
    use crate::memory::types::{MemoryType, Scope};

    fn test_db() -> Connection {
        db::load_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::db::schema::init_schema(&conn).unwrap();
        conn
    }

    fn embedding(dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 384];
        v[dim % 384] = 1.0;
        v
    }

    fn insert(conn: &mut Connection, content: &str, mt: MemoryType, scope: Scope, group: &str, dim: usize) -> String {
        store::store_memory(conn, content, mt, scope, Some(group), 1.0, None, None, &embedding(dim), 0.92)
            .unwrap()
            .id
    }

    #[test]
    fn test_empty_db_stats() {
        let conn = test_db();
        let stats = memory_stats(&conn, None, None).unwrap();
        assert_eq!(stats.total_memories, 0);
        assert_eq!(stats.active_memories, 0);
        assert_eq!(stats.superseded_memories, 0);
        assert_eq!(stats.entity_relations, 0);
        assert_eq!(stats.by_type["episodic"], 0);
        assert_eq!(stats.by_type["semantic"], 0);
        assert!(stats.oldest_memory.is_none());
        assert!(stats.newest_memory.is_none());
    }

    #[test]
    fn test_stats_counts_by_type_and_scope() {
        let mut conn = test_db();
        insert(&mut conn, "Fact one", MemoryType::Semantic, Scope::Global, "default", 0);
        insert(&mut conn, "Fact two", MemoryType::Semantic, Scope::Global, "default", 1);
        insert(&mut conn, "Event one", MemoryType::Episodic, Scope::Group, "default", 2);
        insert(&mut conn, "Entity one", MemoryType::Entity, Scope::Global, "default", 3);

        let stats = memory_stats(&conn, None, None).unwrap();
        assert_eq!(stats.total_memories, 4);
        assert_eq!(stats.active_memories, 4);
        assert_eq!(stats.superseded_memories, 0);
        assert_eq!(stats.by_type["semantic"], 2);
        assert_eq!(stats.by_type["episodic"], 1);
        assert_eq!(stats.by_type["entity"], 1);
        assert_eq!(stats.by_type["procedural"], 0);
        assert_eq!(stats.by_scope["global"], 3);
        assert_eq!(stats.by_scope["group"], 1);
    }

    #[test]
    fn test_stats_with_superseded() {
        let mut conn = test_db();
        let id_old = insert(&mut conn, "Old fact", MemoryType::Semantic, Scope::Global, "default", 0);
        store::store_memory(
            &mut conn, "New fact", MemoryType::Semantic, Scope::Global,
            Some("default"), 1.0, None, Some(&id_old), &embedding(1), 0.92,
        ).unwrap();

        let stats = memory_stats(&conn, None, None).unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.active_memories, 1);
        assert_eq!(stats.superseded_memories, 1);
    }

    #[test]
    fn test_stats_group_filter() {
        let mut conn = test_db();
        insert(&mut conn, "Global fact", MemoryType::Semantic, Scope::Global, "project-a", 0);
        insert(&mut conn, "Group A event", MemoryType::Episodic, Scope::Group, "project-a", 1);
        insert(&mut conn, "Group B event", MemoryType::Episodic, Scope::Group, "project-b", 2);

        let stats = memory_stats(&conn, Some("project-a"), None).unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.by_type["semantic"], 1);
        assert_eq!(stats.by_type["episodic"], 1);
    }

    #[test]
    fn test_stats_timestamps() {
        let mut conn = test_db();
        insert(&mut conn, "First memory", MemoryType::Semantic, Scope::Global, "default", 0);
        insert(&mut conn, "Second memory", MemoryType::Semantic, Scope::Global, "default", 1);

        let stats = memory_stats(&conn, None, None).unwrap();
        assert!(stats.oldest_memory.is_some());
        assert!(stats.newest_memory.is_some());
    }

    #[test]
    fn test_stats_entity_relations_count() {
        let mut conn = test_db();
        let id_a = insert(&mut conn, "Person A", MemoryType::Entity, Scope::Global, "default", 0);
        let id_b = insert(&mut conn, "Person B", MemoryType::Entity, Scope::Global, "default", 1);
        crate::memory::relations::store_relation(&conn, &id_a, "knows", &id_b).unwrap();

        let stats = memory_stats(&conn, None, None).unwrap();
        assert_eq!(stats.entity_relations, 1);
    }
}
