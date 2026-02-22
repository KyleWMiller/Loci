# Loci — Product Requirements Document

> **A memory palace for AI agents.**
>
> Cognitive memory MCP server. Single binary. Local-first. Built in Rust.

---

## Vision

A standalone MCP server that gives any AI agent — Claude Code, Cowork, custom Agent SDK agents, or any MCP-compatible client — persistent, structured, cross-session memory inspired by cognitive science. One memory system, accessible from every surface, owned entirely by the user.

Named for the *method of loci* — the ancient memory palace technique where information is stored by associating it with specific locations in a mental space. Loci builds a structured palace for your AI agents: episodic rooms for what happened, semantic halls for what's known, procedural corridors for how things work, and entity chambers for who and what matters.

The core insight: not all memories are the same kind of thing. Treating them uniformly (flat vector store, growing markdown file) loses structure that matters for retrieval, aging, and relevance. Loci implements a four-type taxonomy — episodic, semantic, procedural, entity — with type-appropriate storage, retrieval, and lifecycle behaviors.

---

## Principles

1. **Simple codebase, real capability.** Single Rust binary. No runtime dependencies. No cloud services required.
2. **Local-first.** All data stays on the user's machine. SQLite for storage, ONNX Runtime for embeddings. No API calls unless explicitly configured.
3. **Progressive disclosure.** Never dump all memories into context. Summaries first, hydrate on demand, respect token budgets.
4. **Anthropic-aligned architecture.** MCP for connectivity. Skills-compatible. Works within Claude Code's security model — no blanket system access.
5. **Portable.** Not locked to Claude Code. Any MCP-compatible client can connect. Memory survives tool changes.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    MCP Clients                        │
│  ┌────────────┐ ┌────────┐ ┌──────┐ ┌────────────┐  │
│  │ Claude Code │ │ Cowork │ │ Custom │ │ Agent SDK  │  │
│  └─────┬──────┘ └───┬────┘ └──┬───┘ └─────┬──────┘  │
│        └────────────┼─────────┼────────────┘         │
│                     │  stdio / SSE                    │
├─────────────────────┼────────────────────────────────┤
│              loci server                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  MCP Tool Interface                             │ │
│  │  store_memory · recall_memory · forget_memory   │ │
│  │  memory_stats · memory_inspect                  │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  Memory Engine                                  │ │
│  │  ┌────────────┐ ┌───────────┐ ┌──────────────┐ │ │
│  │  │ Write Path │ │ Read Path │ │ Maintenance  │ │ │
│  │  │ - classify │ │ - hybrid  │ │ - compaction │ │ │
│  │  │ - dedup    │ │   search  │ │ - decay      │ │ │
│  │  │ - embed    │ │ - RRF     │ │ - promotion  │ │ │
│  │  │ - store    │ │ - budget  │ │ - cleanup    │ │ │
│  │  └────────────┘ └───────────┘ └──────────────┘ │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  Storage Layer                                  │ │
│  │  ┌──────────┐ ┌───────────┐ ┌────────────────┐ │ │
│  │  │ SQLite   │ │ FTS5      │ │ sqlite-vec     │ │ │
│  │  │ memories │ │ keyword / │ │ vector / cosine│ │ │
│  │  │ entities │ │ BM25      │ │ 384-dim        │ │ │
│  │  │ log      │ │           │ │                │ │ │
│  │  └──────────┘ └───────────┘ └────────────────┘ │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  Embedding Engine                               │ │
│  │  ONNX Runtime + all-MiniLM-L6-v2 (384-dim)     │ │
│  │  Local-first · Cached at ~/.loci/models/        │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  Data: ~/.loci/memory.db                             │
│  Config: ~/.loci/config.toml                         │
└──────────────────────────────────────────────────────┘
```

---

## Memory Taxonomy

### Type Definitions

| Type | Human Analogy | What It Stores | Default Scope | Lifecycle |
|------|--------------|----------------|---------------|-----------|
| **Episodic** | "I remember that session" | Past interactions, events, decisions | group | Decays over time; compacts into summaries; may promote to semantic |
| **Semantic** | "I know that X is true" | Facts, knowledge, preferences | global | Long-lived; reinforced by repeated access; updated via supersession |
| **Procedural** | "I know how to do X" | Workflows, action patterns, sequences | global | Stable; versioned via supersession when process changes |
| **Entity** | "I know who/what X is" | People, projects, systems, tools | global | Stable; relationships tracked in entity_relations graph |

### Scope Model

| Scope | Visibility | Use Case |
|-------|-----------|----------|
| **global** | All sessions, all projects | Preferences, entities, workflows |
| **group** | Sessions within a specific project/context | Project-specific decisions, session history |

Group identity is passed via environment variable (`LOCI_GROUP`) or tool parameter. When unset, defaults to `"default"`.

---

## Data Model

### Core Schema

```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,                     -- UUID v7 (time-sortable)
  type TEXT NOT NULL                       -- 'episodic' | 'semantic' | 'procedural' | 'entity'
    CHECK(type IN ('episodic','semantic','procedural','entity')),
  content TEXT NOT NULL,                   -- Natural language memory content
  source_group TEXT,                       -- Group/project where memory originated
  scope TEXT NOT NULL DEFAULT 'global'     -- 'global' | 'group'
    CHECK(scope IN ('global','group')),
  confidence REAL NOT NULL DEFAULT 1.0     -- 0.0-1.0, decays or boosts over time
    CHECK(confidence >= 0.0 AND confidence <= 1.0),
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed TEXT,                      -- ISO 8601
  created_at TEXT NOT NULL,                -- ISO 8601
  updated_at TEXT NOT NULL,                -- ISO 8601
  superseded_by TEXT,                      -- ID of memory that replaces this one
  metadata TEXT                            -- JSON blob for type-specific data
);

CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_memories_scope ON memories(scope);
CREATE INDEX idx_memories_group ON memories(source_group);
CREATE INDEX idx_memories_confidence ON memories(confidence);
CREATE INDEX idx_memories_superseded ON memories(superseded_by);

-- Full-text search (keyword / BM25)
CREATE VIRTUAL TABLE memories_fts USING fts5(
  content,
  id UNINDEXED,
  type UNINDEXED,
  content='memories',
  content_rowid='rowid'
);

-- Vector similarity search (384-dim, cosine distance)
CREATE VIRTUAL TABLE memories_vec USING vec0(
  id TEXT PRIMARY KEY,
  embedding FLOAT[384]
);

-- Entity relationship graph (lightweight triples)
CREATE TABLE entity_relations (
  id TEXT PRIMARY KEY,                     -- UUID
  subject_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  predicate TEXT NOT NULL,                 -- e.g. 'works_at', 'manages', 'part_of'
  object_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL
);

CREATE INDEX idx_relations_subject ON entity_relations(subject_id);
CREATE INDEX idx_relations_object ON entity_relations(object_id);
CREATE INDEX idx_relations_predicate ON entity_relations(predicate);

-- Audit log
CREATE TABLE memory_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL                  -- 'create' | 'update' | 'supersede' | 'decay' | 'compact' | 'delete'
    CHECK(operation IN ('create','update','supersede','decay','compact','delete')),
  memory_id TEXT NOT NULL,
  details TEXT,                            -- JSON
  created_at TEXT NOT NULL
);
```

### Type-Specific Metadata

Stored in the `metadata` JSON column. Not enforced at the schema level — validated at the application layer.

**Episodic:**
```json
{
  "event_date": "2026-02-21",
  "session_id": "sess_abc123",
  "summary": true
}
```

**Semantic:**
```json
{
  "category": "preference",
  "subject": "programming_language"
}
```

**Procedural:**
```json
{
  "trigger": "deployment",
  "steps": ["test", "build", "deploy", "verify"],
  "version": 2
}
```

**Entity:**
```json
{
  "entity_type": "person",
  "name": "John Smith",
  "aliases": ["John", "JS"],
  "properties": {
    "role": "engineering_manager",
    "team": "platform"
  }
}
```

---

## MCP Tools

### `store_memory`

Store a new memory or update an existing one.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | yes | — | Natural language memory content |
| `type` | enum | yes | — | `episodic`, `semantic`, `procedural`, `entity` |
| `scope` | enum | no | by type | `global` or `group`. Defaults: episodic→group, all others→global |
| `metadata` | object | no | `{}` | Type-specific structured data (see above) |
| `supersedes` | string | no | — | ID of memory this replaces. Old memory's `superseded_by` is set. |

**Annotations:** `readOnlyHint: false`, `destructiveHint: false`, `idempotentHint: false`

**Write path behavior:**
1. Embed content using local ONNX model
2. Deduplication check: search same-type memories for cosine similarity > 0.92
   - If match found: update existing memory's `updated_at`, boost confidence by 0.1 (cap 1.0), increment `access_count`. Return existing memory ID with `deduplicated: true`.
   - If no match: insert new memory, FTS entry, and vector embedding
3. If `supersedes` provided: set old memory's `superseded_by` to new ID
4. Write audit log entry
5. Return new memory ID and type

**Response:**
```json
{
  "id": "01953a2b-...",
  "type": "semantic",
  "deduplicated": false,
  "superseded": null
}
```

---

### `recall_memory`

Search and retrieve memories using hybrid search.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | conditional | — | Natural language search query. Required unless `ids` provided. |
| `ids` | string[] | conditional | — | Specific memory IDs to hydrate (progressive disclosure step 2). |
| `type` | enum | no | — | Filter by memory type |
| `scope` | enum | no | — | Filter by scope |
| `group` | string | no | env var | Group to include for group-scoped memories |
| `max_results` | integer | no | 5 | Maximum results to return (1-20) |
| `summary_only` | boolean | no | false | Return compact summaries (id, type, first line, score) instead of full content |
| `token_budget` | integer | no | config | Maximum estimated tokens for response |
| `min_confidence` | float | no | 0.1 | Minimum confidence threshold |

**Annotations:** `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true`

**Read path behavior:**
1. If `ids` provided: direct lookup, skip search, return full content (hydration mode)
2. If `query` provided: run hybrid search
   a. Vector search: embed query, KNN against `memories_vec`
   b. Keyword search: BM25 against `memories_fts`
   c. Merge with Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank)` where k=60
3. Post-merge filters: exclude superseded, apply scope (global + current group), apply type filter, apply min_confidence
4. Sort by RRF score descending
5. Apply token budget: estimate tokens per result (`chars / 4`), include results until budget exhausted
6. Update `access_count` and `last_accessed` for returned memories
7. If `summary_only`: return compact index (id, type, first 80 chars, score)

**Response (full):**
```json
{
  "results": [
    {
      "id": "01953a2b-...",
      "type": "semantic",
      "content": "User prefers Rust over Go for systems programming",
      "confidence": 0.95,
      "score": 0.034,
      "created_at": "2026-02-10T...",
      "metadata": { "category": "preference" }
    }
  ],
  "total_matched": 12,
  "token_estimate": 847
}
```

**Response (summary_only):**
```json
{
  "results": [
    {
      "id": "01953a2b-...",
      "type": "semantic",
      "preview": "User prefers Rust over Go for systems programming",
      "score": 0.034
    }
  ],
  "total_matched": 12,
  "token_estimate": 120
}
```

**Progressive disclosure workflow (taught to agent via CLAUDE.md):**
1. `recall_memory(query: "...", summary_only: true, max_results: 10)` → scan index
2. Agent selects relevant IDs from summaries
3. `recall_memory(ids: ["id1", "id2"])` → hydrate full content for selected memories

---

### `forget_memory`

Mark a memory as superseded or delete it.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | yes | — | ID of memory to forget |
| `reason` | string | no | — | Why this memory is being forgotten |
| `hard_delete` | boolean | no | false | Permanently delete instead of soft-supersede |

**Annotations:** `readOnlyHint: false`, `destructiveHint: true`, `idempotentHint: true`

**Behavior:**
- Default (soft): set `superseded_by` to `"forgotten"`, log reason
- Hard delete: remove from `memories`, `memories_fts`, `memories_vec`, and any `entity_relations`. Log operation.

---

### `memory_stats`

Return statistics about the memory store.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `group` | string | no | — | Filter stats to a specific group |

**Annotations:** `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true`

**Response:**
```json
{
  "total_memories": 342,
  "active_memories": 298,
  "superseded_memories": 44,
  "by_type": {
    "episodic": 156,
    "semantic": 87,
    "procedural": 23,
    "entity": 32
  },
  "by_scope": {
    "global": 201,
    "group": 97
  },
  "entity_relations": 48,
  "db_size_bytes": 2457600,
  "oldest_memory": "2026-01-15T...",
  "newest_memory": "2026-02-21T..."
}
```

---

### `memory_inspect`

Inspect a specific memory's full details including relations and audit history.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | yes | — | ID of memory to inspect |
| `include_relations` | boolean | no | true | Include entity relationships |
| `include_log` | boolean | no | false | Include audit log entries |

**Annotations:** `readOnlyHint: true`, `destructiveHint: false`, `idempotentHint: true`

**Response:**
```json
{
  "memory": {
    "id": "...",
    "type": "entity",
    "content": "John Smith is the engineering manager on the platform team",
    "confidence": 0.95,
    "access_count": 12,
    "metadata": { "entity_type": "person", "name": "John Smith" }
  },
  "relations": [
    { "predicate": "manages", "object": { "id": "...", "preview": "Platform team" } },
    { "predicate": "works_at", "object": { "id": "...", "preview": "Acme Corp" } }
  ],
  "log": []
}
```

---

### `store_relation`

Create a relationship between two entity memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `subject_id` | string | yes | — | Source entity memory ID |
| `predicate` | string | yes | — | Relationship type (e.g. `works_at`, `manages`, `part_of`, `related_to`) |
| `object_id` | string | yes | — | Target entity memory ID |

**Annotations:** `readOnlyHint: false`, `destructiveHint: false`, `idempotentHint: true`

**Behavior:** Validates both IDs exist and are entity-type memories. Creates relationship. Deduplicates on (subject, predicate, object) tuple.

---

## Embedding Engine

### Model

| Property | Value |
|----------|-------|
| Model | `all-MiniLM-L6-v2` (ONNX format) |
| Dimensions | 384 |
| Runtime | `ort` crate (ONNX Runtime Rust bindings) |
| Model location | `~/.loci/models/all-MiniLM-L6-v2.onnx` |
| Model size | ~30 MB |
| Tokenizer | Included HuggingFace tokenizer JSON |

### Behavior

- Model is downloaded on first use and cached locally
- All embedding happens locally — no network calls
- Batch embedding supported for bulk operations (compaction, migration)
- Fallback configuration available for API-based providers (Voyage AI, OpenAI) via config

### Configuration

```toml
[embedding]
provider = "local"                              # "local" | "voyage" | "openai"
model = "all-MiniLM-L6-v2"
cache_dir = "~/.loci/models"

# Optional API fallback
[embedding.api]
provider = "voyage"
model = "voyage-3.5"
api_key_env = "VOYAGE_API_KEY"                  # Read from environment variable
```

---

## Memory Maintenance

### Confidence Decay

Memories that are never accessed gradually lose relevance.

- **Trigger:** Periodic maintenance (configurable interval, default weekly)
- **Formula:** `new_confidence = confidence * decay_factor`
- **Decay factor:** 0.95 per interval for episodic, 0.99 for semantic/procedural/entity
- **Access boost:** Each recall boosts confidence by 0.05 (capped at 1.0)
- **Floor:** Memories below 0.1 confidence are flagged for cleanup but not auto-deleted

### Compaction

Consolidate old episodic memories into summaries.

- **Trigger:** Periodic maintenance
- **Criteria:** Episodic memories older than configurable threshold (default 30 days), grouped by week, minimum 5 per group
- **Process:**
  1. Group qualifying episodic memories by week and source_group
  2. Concatenate content, generate a summary (using the LLM via a compaction prompt, or simple concatenation if no LLM available)
  3. Create new episodic memory with `metadata.summary: true`
  4. Mark originals as superseded by the summary
  5. Log compaction operation

### Episodic-to-Semantic Promotion

Extract durable facts from repeated episodic observations.

- **Trigger:** During compaction
- **Criteria:** Same fact (cosine similarity > 0.88) appears in 3+ episodic memories
- **Process:**
  1. Identify clusters of similar episodic content
  2. Create semantic memory with the distilled fact
  3. Do NOT supersede the episodic memories (they retain event context)
  4. Log promotion operation

### Cleanup

- Memories with confidence < 0.05 and no access in 90+ days are candidates for hard deletion
- Cleanup is manual by default (via CLI command), never automatic
- Audit log retains deletion records

---

## Configuration

### Config File: `~/.loci/config.toml`

```toml
[server]
transport = "stdio"                             # "stdio" | "sse"
log_level = "info"                              # "error" | "warn" | "info" | "debug" | "trace"

[storage]
db_path = "~/.loci/memory.db"
default_group = "default"

[embedding]
provider = "local"
model = "all-MiniLM-L6-v2"
cache_dir = "~/.loci/models"

[retrieval]
default_max_results = 5
preload_token_budget = 2000                     # Budget for summary-mode recalls
recall_token_budget = 4000                      # Budget for full hydration recalls
rrf_k = 60                                      # RRF constant
dedup_threshold = 0.92                          # Cosine similarity for dedup gate

[maintenance]
enabled = false                                 # Disabled by default until stable
interval_days = 7
episodic_decay_factor = 0.95
semantic_decay_factor = 0.99
compaction_age_days = 30
compaction_min_group_size = 5
promotion_threshold = 3                         # Episodic occurrences before promoting to semantic
promotion_similarity = 0.88
cleanup_confidence_floor = 0.05
cleanup_no_access_days = 90
```

### Environment Variables

| Variable | Purpose | Overrides |
|----------|---------|-----------|
| `LOCI_DB` | Database path | `storage.db_path` |
| `LOCI_GROUP` | Current group/project context | `storage.default_group` |
| `LOCI_LOG_LEVEL` | Log verbosity | `server.log_level` |
| `VOYAGE_API_KEY` | Voyage AI API key (if using API embeddings) | — |
| `OPENAI_API_KEY` | OpenAI API key (if using API embeddings) | — |

---

## Client Integration

### Claude Code

**.mcp.json (project-level or global):**
```json
{
  "mcpServers": {
    "loci": {
      "command": "loci",
      "args": ["serve"],
      "env": {
        "LOCI_GROUP": "${workspaceFolder}"
      }
    }
  }
}
```

**CLAUDE.md guidance for the agent:**
```markdown
## Memory

You have access to a persistent memory system (`loci` MCP tools).
Use it proactively throughout every session.

### When to store memories:
- User states a preference → `store_memory` type: semantic
- You learn about a person, project, or system → `store_memory` type: entity
- You learn or execute a multi-step workflow → `store_memory` type: procedural
- A significant decision is made or event occurs → `store_memory` type: episodic

### When to recall memories:
- At session start: `recall_memory(query: "<topic of this session>", summary_only: true)`
- Before making assumptions about user preferences
- When the user references something from a past session
- When you need context about a person, project, or system

### Progressive disclosure:
1. First call: `recall_memory(query: "...", summary_only: true, max_results: 10)`
2. Scan summaries, identify relevant IDs
3. Second call: `recall_memory(ids: ["relevant_id_1", "relevant_id_2"])`

### Session exit:
Before ending a significant session, store an episodic summary:
`store_memory(content: "Session summary: ...", type: "episodic")`

### Entity relationships:
When you identify relationships between entities, use `store_relation`:
`store_relation(subject_id: "person_id", predicate: "works_at", object_id: "company_id")`
```

---

## CLI Interface

The binary serves dual purpose: MCP server and management CLI.

```
loci serve                     # Start MCP server (stdio transport)
loci serve --sse               # Start MCP server (SSE transport, for remote)

loci search <query>            # Interactive search from terminal
loci stats                     # Memory statistics
loci inspect <id>              # Full memory details
loci export                    # Export all memories as JSON
loci import <file>             # Import memories from JSON
loci compact                   # Run compaction manually
loci cleanup --dry-run         # Preview what would be cleaned up
loci cleanup                   # Delete low-confidence stale memories
loci reset                     # Delete all memories (requires confirmation)
loci model download            # Pre-download the embedding model
```

---

## Rust Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `rusqlite` | SQLite with bundled feature, extension loading |
| `sqlite-vec` | Vector similarity search (loaded as SQLite extension) |
| `ort` | ONNX Runtime for local embeddings |
| `tokenizers` | HuggingFace tokenizer for MiniLM input processing |
| `mcp-rust-sdk` or `rmcp` | MCP protocol implementation (evaluate at build time) |
| `uuid` | UUID v7 generation |
| `serde` / `serde_json` | Serialization |
| `toml` | Config file parsing |
| `clap` | CLI argument parsing |
| `tracing` | Structured logging |
| `tokio` | Async runtime (for SSE transport, model download) |
| `reqwest` | HTTP client (model download, optional API embeddings) |
| `dirs` | Platform-appropriate home directory resolution |

---

## Project Structure

```
loci/
├── Cargo.toml
├── README.md
├── LICENSE
├── config.example.toml
├── src/
│   ├── main.rs                    # CLI entry point (clap), dispatches to serve/commands
│   ├── server.rs                  # MCP server setup, tool registration, transport
│   ├── config.rs                  # Config loading (toml file + env vars)
│   ├── db/
│   │   ├── mod.rs                 # DB init, migrations, sqlite-vec loading
│   │   ├── schema.rs              # SQL schema definitions
│   │   └── migrations.rs          # Schema versioning
│   ├── memory/
│   │   ├── mod.rs                 # Public API surface
│   │   ├── types.rs               # MemoryType, Memory, Scope, Relation enums/structs
│   │   ├── store.rs               # Write path: create, update, supersede, dedup gate
│   │   ├── search.rs              # Read path: hybrid search, RRF merge, token budgeting
│   │   ├── relations.rs           # Entity relationship CRUD
│   │   └── maintenance.rs         # Decay, compaction, promotion, cleanup
│   ├── embedding/
│   │   ├── mod.rs                 # Trait definition, provider dispatch
│   │   ├── local.rs               # ONNX Runtime + MiniLM pipeline
│   │   └── api.rs                 # Optional: Voyage/OpenAI API fallback
│   ├── tools/
│   │   ├── mod.rs                 # Tool registration
│   │   ├── store_memory.rs        # MCP tool handler
│   │   ├── recall_memory.rs       # MCP tool handler
│   │   ├── forget_memory.rs       # MCP tool handler
│   │   ├── memory_stats.rs        # MCP tool handler
│   │   ├── memory_inspect.rs      # MCP tool handler
│   │   └── store_relation.rs      # MCP tool handler
│   └── cli/
│       ├── mod.rs                 # CLI command dispatch
│       ├── search.rs              # Terminal search command
│       ├── stats.rs               # Stats display
│       ├── export.rs              # JSON export
│       ├── import.rs              # JSON import
│       └── maintenance.rs         # Manual compact/cleanup commands
└── tests/
    ├── integration/
    │   ├── store_test.rs          # Write path tests
    │   ├── search_test.rs         # Hybrid search tests
    │   ├── dedup_test.rs          # Deduplication gate tests
    │   ├── maintenance_test.rs    # Decay, compaction, promotion tests
    │   └── mcp_test.rs            # End-to-end MCP tool tests
    └── fixtures/
        └── seed_memories.json     # Test data
```

---

## Feature Milestones

### M0: Foundation

Establish the core infrastructure. After this milestone, the binary compiles, the database initializes, and the MCP server starts and responds to tool calls with stub responses.

- Project scaffolding: Cargo workspace, clap CLI, config loading
- SQLite database initialization with full schema (memories, FTS5, entity_relations, memory_log)
- sqlite-vec extension loading and vec0 virtual table creation
- MCP server with stdio transport, tool registration (stubs)
- `loci serve` and `loci model download` commands

### M1: Embedding Engine

Local embedding pipeline operational. After this milestone, text can be converted to 384-dim vectors without any network calls.

- ONNX Runtime integration via `ort` crate
- MiniLM-L6-v2 model download and caching to `~/.loci/models/`
- Tokenizer loading (HuggingFace tokenizer JSON)
- `embed(text) -> Vec<f32>` function with batch support
- Unit tests for embedding consistency and dimension validation

### M2: Write Path

Memories can be stored. After this milestone, `store_memory` is fully functional including deduplication.

- `store_memory` MCP tool handler (full implementation)
- Memory creation: UUID v7 generation, embedding, SQLite insert, FTS5 sync, vec0 insert
- Deduplication gate: cosine similarity check against same-type memories, threshold 0.92
- Supersession: `superseded_by` linkage when `supersedes` param provided
- Audit log writes on every mutation
- Scope defaulting by memory type

### M3: Read Path

Memories can be retrieved. After this milestone, `recall_memory` delivers hybrid search results with progressive disclosure.

- `recall_memory` MCP tool handler (full implementation)
- Vector similarity search via sqlite-vec KNN
- BM25 keyword search via FTS5
- Reciprocal Rank Fusion merge (k=60)
- Post-merge filtering: superseded exclusion, scope rules, type filter, confidence floor
- Token budget enforcement (`chars / 4` estimation)
- Summary-only mode for progressive disclosure step 1
- ID-based hydration for progressive disclosure step 2
- Access tracking: bump `access_count` and `last_accessed` on retrieval

### M4: Entity Graph

Entity relationships can be stored and queried. After this milestone, the lightweight knowledge graph is functional.

- `store_relation` MCP tool handler
- `memory_inspect` tool with relation traversal
- Relation deduplication on (subject, predicate, object)
- Cascade behavior: relations cleaned up when entities are forgotten
- Entity-aware search: when recalling an entity, include its direct relations in results

### M5: Management Tools

Introspection and management via MCP tools and CLI. After this milestone, users can understand and manage their memory store.

- `memory_stats` MCP tool handler
- `forget_memory` MCP tool handler (soft and hard delete)
- CLI commands: `search`, `stats`, `inspect`, `export`, `import`, `reset`
- JSON export/import format for backup and migration

### M6: Maintenance Engine

Automated memory lifecycle management. After this milestone, the memory store self-maintains over time.

- Confidence decay (per-type decay factors)
- Confidence boost on access
- Episodic compaction (age threshold, weekly grouping, summary generation)
- Episodic-to-semantic promotion (occurrence threshold, similarity clustering)
- Cleanup command (confidence floor + staleness threshold, dry-run support)
- CLI: `compact`, `cleanup --dry-run`, `cleanup`
- Configuration for all maintenance parameters

### M7: Resilience and Polish

Production hardening. After this milestone, the tool is reliable enough for daily use.

- Schema migrations (versioned, forward-only)
- Graceful handling of corrupt/missing DB
- Concurrent access safety (SQLite WAL mode, connection pooling)
- Embedding model version tracking (re-embed on model change)
- SSE transport option for remote server deployment
- Comprehensive integration tests
- README, installation docs, CLAUDE.md template

### M8: Advanced Retrieval (Post-Launch)

Enhanced search capabilities based on real-world usage patterns.

- Temporal search: "what did we discuss last week"
- Relation-aware search: traverse entity graph during recall
- Memory clustering: group related memories for context-rich retrieval
- API embedding fallback (Voyage AI, OpenAI) when local model insufficient
- Optional web viewer UI (localhost, similar to claude-mem's)

---

## Distribution

### Primary: Cargo Install

```bash
cargo install loci
```

Single binary, no runtime dependencies. The ONNX model downloads on first use.

### Secondary: Pre-built Binaries

GitHub Releases with binaries for:
- `x86_64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin` (Apple Silicon)
- `x86_64-pc-windows-msvc`

### Tertiary: Homebrew

```bash
brew install loci
```

---

## Success Criteria

- Agent consistently stores relevant memories without prompting (CLAUDE.md is sufficient guidance)
- Recall latency < 200ms for hybrid search against 1,000 memories
- Token overhead for memory pre-loading stays within configured budget
- Zero network calls during normal operation (local embeddings)
- Memory store grows manageably (compaction prevents unbounded growth)
- Portable: works with Claude Code, Cowork, and any MCP-compatible client
- Single binary install, no Docker, no Node.js, no Python required

---

## Non-Goals (Explicit Exclusions)

- **Multi-user / shared memory** — This is a personal memory system. Shared team memory is a different product.
- **Real-time sync** — No cloud sync, no cross-device replication. Memory lives on one machine.
- **LLM-in-the-loop for writes** — The agent decides what to store. No separate summarization LLM call on the write path (except during compaction, which is offline).
- **Full knowledge graph** — The entity_relations table is intentionally lightweight. Not building a graph database.
- **Model training / fine-tuning** — Memory is retrieval-only, not used for model adaptation.
- **Crypto token / monetization** — This is a tool, not a platform.

---

## Open Questions

1. **Rust MCP SDK maturity:** Evaluate `mcp-rust-sdk` vs `rmcp` vs raw protocol implementation at M0. The MCP stdio protocol is simple enough that a minimal custom implementation may be more maintainable than depending on an immature crate.

2. **Compaction without LLM:** M6 compaction ideally uses an LLM to summarize grouped episodic memories. Without an LLM available, fallback to concatenation with truncation. Consider whether the agent itself should be prompted to compact (via a `compact_memories` tool) rather than the server doing it autonomously.

3. **sqlite-vec in Rust:** The `sqlite-vec` npm package has Node.js bindings. For Rust, load the C extension via `rusqlite::Connection::load_extension`. Verify the extension binary ships correctly across platforms or bundle it.

4. **Embedding model distribution:** The MiniLM ONNX model (~30MB) needs to be downloaded on first use. Consider bundling it in the binary (increases binary size to ~45MB) vs. lazy download. Lazy download is more flexible but requires network on first run.

5. **Group identity in non-Claude-Code contexts:** Claude Code can pass `${workspaceFolder}` as the group. For Cowork, Agent SDK, or other clients, the group needs to be set explicitly. Define conventions for group naming to avoid fragmentation.
