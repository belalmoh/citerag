# CiteRAG — Production-Grade RAG Document System

## Project Overview
A production-grade RAG (Retrieval-Augmented Generation) document intelligence system. Users upload documents (PDF, DOCX, TXT, etc.), which are parsed, chunked, embedded, and indexed into Qdrant. Users then ask questions, and the system retrieves relevant chunks via hybrid search (dense + keyword with RRF fusion), re-ranks them with a cross-encoder, and generates cited answers via an LLM.

## Architecture
Upload → Parse → Chunk → Embed → Index (Qdrant)
↓
Query → Embed → Hybrid Search → Rerank → Generate (LLM) → Stream Response



## Tech Stack
- **API**: FastAPI (async-native)
- **Vector DB**: Qdrant (self-hosted, hybrid search built-in)
- **Relational DB**: PostgreSQL 16 (async via asyncpg + SQLAlchemy 2.0)
- **Task Queue**: Celery + Redis
- **Embedding**: OpenAI-compatible (Ollama in dev) + sentence-transformers fallback
- **LLM**: OpenAI-compatible (Ollama in dev)
- **Re-ranker**: sentence-transformers CrossEncoder
- **Parser**: pymupdf (PDF, DOCX, EPUB, XPS, etc.)

## Project Structure
citerag/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── api/                 # FastAPI routers
│   │   ├── upload.py        # POST /upload
│   │   ├── query.py         # POST /query + /query/stream
│   │   ├── feedback.py      # POST /feedback
│   │   ├── jobs.py          # GET /jobs/{id}
│   │   └── health.py        # GET /health
│   ├── core/                # Shared infrastructure
│   │   ├── config.py        # Pydantic Settings
│   │   ├── database.py      # Async Postgres session
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── dependencies.py  # DI singletons
│   ├── ingestion/           # Document processing pipeline
│   │   ├── parser.py        # pymupdf document parser
│   │   ├── chunker.py       # Structure-aware chunking
│   │   ├── embedder.py      # Embedding (OpenAI + local + fallback)
│   │   ├── indexer.py       # Qdrant upsert/delete
│   │   └── worker.py        # Celery task definitions
│   ├── retrieval/           # Query-time retrieval
│   │   ├── retriever.py     # Basic vector search
│   │   ├── hybrid_search.py # Dense + keyword + RRF fusion
│   │   ├── reranker.py      # Cross-encoder re-ranker
│   │   └── query_expander.py # Query expansion / HyDE
│   ├── generation/          # LLM response generation
│   │   ├── llm_client.py    # OpenAI-compatible client
│   │   ├── prompt_builder.py # Citation-aware prompts
│   │   └── stream_handler.py # SSE streaming
│   ├── evaluation/          # RAG evaluation
│   │   ├── ragas_eval.py
│   │   ├── llm_judge.py
│   │   └── test_queries.py
│   └── utils/               # Shared utilities
│       ├── file_storage.py
│       ├── logging_config.py
│       └── validators.py
├── tests/
├── scripts/
├── notebooks/
└── docker-compose.yml       # Postgres + Qdrant + Redis



## Build Order (Step-by-Step Learning Path)

### Phase 1: Core Infrastructure (DONE)
1. **Config & Database** — Pydantic Settings, async Postgres engine, session factory
2. **SQLAlchemy Models** — Document, Chunk, QueryLogs, Feedback with relationships
3. **Pydantic Schemas** — Request/response models for API
4. **Dependency Injection** — Singleton factories for shared services

### Phase 2: Ingestion Pipeline (DONE)
5. **Document Parser** — pymupdf wrapper (sync + async), table extraction
6. **Chunker** — RecursiveCharacterTextSplitter + protected regions (tables, code, lists)
7. **Embedder** — Abstract base + OpenAI + Local + FallbackEmbedder
8. **Indexer** — Qdrant collection management, upsert, delete
9. **Worker** — Celery task orchestrating parse → chunk → embed → index

### Phase 3: API Endpoints (DONE)
10. **Upload** — File receive → DB insert → Celery enqueue
11. **Jobs** — Poll ingestion status by document ID
12. **Health** — Simple health check

### Phase 4: Retrieval Pipeline (IN PROGRESS — ~75%)
13. **Basic Retriever** — Vector search via Qdrant `query_points` (DONE)
14. **Hybrid Search** — Dense + keyword with RRF fusion (DONE, uncommitted)
15. **Re-ranker** — Cross-encoder scoring (DONE)
16. **Query Expander** — LLM-based query paraphrasing (NOT STARTED)

### Phase 5: Generation (DONE)
17. **LLM Client** — OpenAI-compatible with retry + streaming
18. **Prompt Builder** — Citation-aware prompt templates
19. **Stream Handler** — SSE streaming with DB logging

### Phase 6: Query Endpoint (DONE)
20. **POST /query** — Full RAG pipeline: embed → hybrid search → rerank → generate → log
21. **POST /query/stream** — SSE streaming variant

### Phase 7: Feedback (DONE)
22. **POST /feedback** — Rate answers, store in Feedback table

### Phase 8: Evaluation (NOT STARTED)
23. **RAGAS Evaluation** — Faithfulness, answer relevancy, context precision/recall
24. **LLM-as-a-Judge** — Automated answer quality scoring
25. **Test Queries** — Curated dataset with known-good answers

### Phase 9: Testing & Tooling (NOT STARTED)
26. **Tests** — Real test coverage for all components
27. **Scripts** — Seed data, eval runner, benchmark
28. **Utilities** — File storage, logging config, validators

## Key Design Decisions

### Why pymupdf instead of Unstructured.io?
The architecture doc specified Unstructured.io, but pymupdf is lighter, faster, handles PDF/DOCX/EPUB natively, and doesn't require external API calls. It's a pragmatic choice for a self-hosted system.

### Why Qdrant's `query_points` with `prefetch` + `FusionQuery(RRF)`?
Qdrant's native hybrid search handles both dense vector search and full-text keyword search in a single API call. RRF (Reciprocal Rank Fusion) combines rankings without needing score normalization. The `prefetch` parameter allows specifying different limits for each search strategy before fusion.

### Why Celery instead of FastAPI BackgroundTasks?
Document ingestion can take 30-60s for large PDFs. Celery provides retry logic, task monitoring, and survives server restarts. FastAPI BackgroundTasks would block the event loop for long-running CPU-bound work.

### Why SSE instead of WebSockets for streaming?
SSE is simpler (standard HTTP, works through proxies, auto-reconnect), sufficient for unidirectional server→client streaming, and doesn't require WebSocket handshake overhead.

### Why `asyncio.to_thread` for pymupdf and CrossEncoder?
Both pymupdf (C++ bindings) and sentence-transformers (PyTorch) block the GIL. Offloading to a thread pool keeps the async event loop responsive.

### Why a dedicated `SearchResult` Pydantic model?
It provides type safety across the retrieval pipeline (retriever → reranker → prompt builder), ensures consistent field names, and makes the data flow explicit.

## Common Pitfalls

1. **Qdrant `query_points` API**: The old `_client.search()` is deprecated. Use `query_points()` with `query=` for vector search or `prefetch=` for hybrid search. The return type is `QueryResponse`, not a raw list.

2. **SQLAlchemy relationship config**: `back_populates` must match on both sides of a relationship. Using `backref` on one side and `back_populates` on the other causes silent failures.

3. **Celery + async boundary**: Celery workers are synchronous. Use `asyncio.run()` inside the task to bridge to async code. Create fresh connections inside the task — don't share event loops or clients across tasks.

4. **Embedding dimension mismatch**: When switching embedding models, the Qdrant collection's vector dimension is fixed at creation. You must delete and recreate the collection, or use a new collection name.

5. **CrossEncoder thread blocking**: `CrossEncoder.predict()` blocks the event loop. Always wrap it in `asyncio.to_thread()`.

6. **File upload race condition**: Save the file to disk first, then insert the DB record, then enqueue the Celery task. If any step fails, the previous steps are already done and can be cleaned up independently.

7. **Reranker model download**: The first call to `CrossEncoder(model_name)` downloads the model from Hugging Face. This can take 10-30s and may fail in air-gapped environments. Consider pre-downloading in Docker build.

## Current Session Context (2026-07-09)

**What's in progress**: Hybrid search implementation. The `hybrid_search.py` file is newly created but uncommitted. The `query.py` endpoint has been updated to use `HybridRetriever` instead of `Retriever`. The `indexer.py` has a new `create_payload_index` for full-text search on the `content` field.

**Next step**: Commit the hybrid search changes, then implement `query_expander.py` for LLM-based query paraphrasing.

## Running the Project
```bash
# Start infrastructure
docker compose up -d

# Run the API
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_chunker.py -v
```

## Environment Variables
See .env.example for all config. Key ones:

OPENAI_API_KEY — API key (can be dummy for local Ollama)
OPENAI_BASE_URL — Defaults to http://localhost:11434/v1 (Ollama)
OPENAI_EMBEDDING_MODEL — Defaults to nomic-embed-text:latest
OPENAI_CHAT_MODEL — Defaults to gemma4:31b-cloud
QDRANT_URL — Defaults to http://localhost:6333
DATABASE_URL — Defaults to local Postgres