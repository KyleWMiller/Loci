#![doc(html_logo_url = "https://raw.githubusercontent.com/KyleWMiller/Loci/main/assets/Loci-logo.png")]

//! <div align="center">
//!   <img src="https://raw.githubusercontent.com/KyleWMiller/Loci/main/assets/Loci-logo.png" width="200" alt="Loci logo" />
//! </div>
//!
//! Cognitive memory for AI agents — persistent, structured, cross-session memory via MCP.
//!
//! Loci is an [MCP](https://modelcontextprotocol.io/) server that gives AI agents a memory
//! system inspired by cognitive science. Memories are stored in four types, each with
//! different scoping, decay rates, and lifecycle behaviors:
//!
//! | Type | Purpose | Default Scope | Decay |
//! |------|---------|---------------|-------|
//! | **Episodic** | Events, decisions, session logs | Group | Fast (0.95/cycle) |
//! | **Semantic** | Facts, knowledge, preferences | Global | Slow (0.99/cycle) |
//! | **Procedural** | Workflows, patterns, how-to | Global | Slow (0.99/cycle) |
//! | **Entity** | People, places, projects | Global | Slow (0.99/cycle) |
//!
//! # Architecture
//!
//! - **Storage**: SQLite with FTS5 for keyword search and
//!   [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search
//! - **Embeddings**: Local ONNX Runtime with all-MiniLM-L6-v2 (384 dimensions)
//! - **Search**: Hybrid vector + BM25 keyword search merged via Reciprocal Rank Fusion
//! - **Transport**: MCP over stdio (primary) or Streamable HTTP/SSE
//!
//! # Modules
//!
//! - [`config`] — Configuration loading from TOML files and environment variables
//! - [`db`] — SQLite database initialization, schema, migrations, and health checks
//! - [`embedding`] — Text-to-vector embedding pipeline via ONNX Runtime
//! - [`memory`] — Core memory engine: store, search, forget, relations, and maintenance

pub mod config;
pub mod db;
pub mod embedding;
pub mod memory;
