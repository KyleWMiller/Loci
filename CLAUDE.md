# Loci — Build Context

## What This Is

Loci is a cognitive memory MCP server built in Rust. It gives AI agents persistent, structured, cross-session memory using a four-type taxonomy (episodic, semantic, procedural, entity) inspired by cognitive science. Named for the "method of loci" (memory palace technique).

**Read `docs/prd.md` before starting any work.** It is the single source of truth for architecture, schema, tools, and milestones.

## Key Design Decisions

- **Single Rust binary** — no runtime dependencies, no Docker, no Node.js
- **MCP server via stdio transport** — clients connect via `.mcp.json` config
- **SQLite + FTS5 + sqlite-vec** — all storage in `~/.loci/memory.db`
- **Local ONNX embeddings** — `all-MiniLM-L6-v2` via `ort` crate, 384-dim, cached at `~/.loci/models/`
- **Hybrid search** — vector similarity + BM25 keyword, merged with Reciprocal Rank Fusion (k=60)
- **Progressive disclosure** — summary-only mode for token-efficient scanning, then hydrate by ID
- **Config** — TOML at `~/.loci/config.toml`, overridable via env vars (`LOCI_MEMORY_DB`, `LOCI_MEMORY_GROUP`, `LOCI_LOG_LEVEL`)

## Architecture at a Glance

```
MCP Client (Claude Code, etc.)
  ↕ stdio
Loci MCP Server
  ├── Tools: store_memory, recall_memory, forget_memory, memory_stats, memory_inspect, store_relation
  ├── Memory Engine: write path (embed → dedup → store), read path (hybrid search → RRF → budget)
  ├── Storage: rusqlite (SQLite + FTS5 + sqlite-vec)
  └── Embeddings: ort (ONNX Runtime) + all-MiniLM-L6-v2
```

## Build Order (from PRD milestones)

Follow this sequence — each milestone builds on the previous:

1. **M0 Foundation** ✅ — Cargo scaffold, clap CLI, config loading, SQLite schema init, sqlite-vec extension loading, MCP server with stub tools
2. **M1 Embedding Engine** ✅ — `ort` integration, model download/caching, `embed(text) -> Vec<f32>`
3. **M2 Write Path** ✅ — `store_memory` tool: UUID v7, embedding, SQLite insert, FTS5 sync, vec0 insert, dedup gate (cosine > 0.92), supersession, audit log
4. **M3 Read Path** ✅ — `recall_memory` tool: vector KNN, BM25 FTS, RRF merge, post-filters, token budgeting, summary_only mode, ID hydration, access tracking
5. **M4 Entity Graph** ✅ — `store_relation` tool, `memory_inspect` with relation traversal, cascade on delete
6. **M5 Management** ✅ — `memory_stats`, `forget_memory`, CLI commands (search, stats, inspect, export, import, reset)
7. **M6 Maintenance** ✅ — confidence decay, compaction, episodic-to-semantic promotion, cleanup
8. **M7 Resilience** — migrations, WAL mode, corrupt DB handling, SSE transport, integration tests

## Critical Implementation Details

### sqlite-vec in Rust
Load as a SQLite extension via `rusqlite::Connection::load_extension`. The `sqlite-vec` crate provides the extension binary. You may need to use the `sqlite_vec::sqlite3_vec_init` function pointer approach with `rusqlite`'s `db.load_extension_enable()` / `db.load_extension()`. Test this early in M0 — if it doesn't work cleanly, fall back to bundling the C extension.

### MCP Protocol in Rust
Evaluate these options at M0:
- `rmcp` crate — most active Rust MCP SDK
- `mcp-rust-sdk` — alternative
- Raw implementation — MCP stdio is JSON-RPC over stdin/stdout, simple enough to implement directly if SDKs are immature

The protocol is: read JSON-RPC from stdin, write JSON-RPC to stdout. Tools are registered with name, description, inputSchema (JSON Schema). Tool calls receive params, return content blocks.

### Embedding Pipeline
```
text → HuggingFace tokenizer (tokenizers crate) → token IDs → ONNX model (ort crate) → 384-dim f32 vector → normalize → store in sqlite-vec
```
The model file is `all-MiniLM-L6-v2.onnx` (~30MB). Download from HuggingFace on first use via `loci model download`, cache to `~/.loci/models/`. The tokenizer JSON file ships alongside the ONNX model.

### Deduplication Gate
Before storing any memory, embed the content and check existing memories of the same type for cosine similarity > 0.92. If match found, update existing memory (bump `updated_at`, boost confidence by 0.1, increment `access_count`) instead of creating a duplicate.

### Token Budget
Token estimation uses `chars / 4` — no tokenizer dependency needed. The ~20% error margin is managed by budgeting conservatively. `recall_memory` stops adding results once the estimated token count exceeds the configured budget.

### RRF Merge
```
score(doc) = Σ 1/(k + rank_in_list)  where k = 60
```
Run vector search and FTS search in parallel, each returns a ranked list, merge scores across lists, sort descending.

### UUID v7
Use the `uuid` crate with v7 support — these are time-sortable, which makes chronological queries efficient.

## Project Structure

```
loci/
├── Cargo.toml
├── CLAUDE.md                      ← you are here
├── README.md
├── LICENSE
├── docs/
│   └── prd.md                     ← the full PRD (source of truth)
├── config.example.toml
├── src/
│   ├── main.rs                    # clap CLI entry point
│   ├── server.rs                  # MCP server setup + tool registration
│   ├── config.rs                  # TOML config + env var loading
│   ├── db/
│   │   ├── mod.rs                 # DB init, sqlite-vec loading
│   │   ├── schema.rs              # SQL CREATE statements
│   │   └── migrations.rs          # Schema versioning
│   ├── memory/
│   │   ├── mod.rs                 # Public API
│   │   ├── types.rs               # MemoryType, Memory, Scope enums/structs
│   │   ├── store.rs               # Write path + dedup
│   │   ├── search.rs              # Hybrid search + RRF
│   │   ├── relations.rs           # Entity relation CRUD
│   │   └── maintenance.rs         # Decay, compaction, promotion
│   ├── embedding/
│   │   ├── mod.rs                 # Trait + provider dispatch
│   │   ├── local.rs               # ONNX Runtime pipeline
│   │   └── api.rs                 # Optional API fallback
│   ├── tools/
│   │   ├── mod.rs                 # Tool registration
│   │   ├── store_memory.rs
│   │   ├── recall_memory.rs
│   │   ├── forget_memory.rs
│   │   ├── memory_stats.rs
│   │   ├── memory_inspect.rs
│   │   └── store_relation.rs
│   └── cli/
│       ├── mod.rs
│       ├── search.rs
│       ├── stats.rs
│       ├── export.rs
│       ├── import.rs
│       └── maintenance.rs
└── tests/
    ├── integration/
    └── fixtures/
```

## Coding Conventions

- Use `anyhow` for error handling in the binary, `thiserror` for library errors
- Use `tracing` for structured logging (not `log` or `println!`)
- Use `serde` + `serde_json` for all serialization
- Use `clap` derive API for CLI args
- Write integration tests that create temp databases — never test against `~/.loci/`
- Keep modules focused — one file per MCP tool handler
- SQL queries as `const &str` in the module that uses them, not a separate queries file

## Versioning

Loci follows semver. The version in `Cargo.toml` must be bumped as part of any change.

**Pre-1.0 rules** (current: 0.x.y):
- **0.MINOR bump** (e.g., 0.1.0 → 0.2.0): Breaking changes — MCP tool schema changes (renamed/removed params, changed response shape), DB schema changes requiring migration, config format changes, CLI arg changes
- **0.x.PATCH bump** (e.g., 0.1.0 → 0.1.1): Non-breaking changes — bug fixes, new features with additive-only API changes, internal refactors, new CLI commands, performance improvements, documentation

When making a change, bump the version in `Cargo.toml` and note the classification in the commit message (e.g., `[patch]` or `[minor]`).

## M0 Priority: sqlite-vec Spike

Before building anything else in M0, validate that sqlite-vec loads and works in Rust. This is the highest-risk integration point. Write a throwaway test that:

1. Opens a rusqlite connection
2. Loads the sqlite-vec extension
3. Creates a `vec0` virtual table with `FLOAT[384]`
4. Inserts a test vector
5. Runs a KNN query with `MATCH`
6. Returns results

If this fails, investigate these fallback approaches in order:
- Use `sqlite_vec::sqlite3_vec_init` as a function pointer with `rusqlite::ffi`
- Bundle the sqlite-vec `.so`/`.dylib`/`.dll` and load via `db.load_extension()`
- Use the `sqlite-vec` crate's `load` function if it provides one for rusqlite

Do not proceed to M1 until vector search works end-to-end in a test.

## What NOT to Do

- Don't add a web UI yet — that's M8 at earliest
- Don't add cloud sync — this is local-first by design
- Don't add multi-user support — personal memory only
- Don't use a separate vector database (Qdrant, Chroma, etc.) — sqlite-vec handles this
- Don't add a crypto token — this is a tool, not a platform
- Don't over-engineer the MCP transport — stdio is the primary target, SSE is M7
