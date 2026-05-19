# CiteRAG — Production-Grade RAG Document System

## Quick Start

1. Copy environment file and fill in your keys:
   ```bash
   cp .env.example .env
   ```

2. Start infrastructure:
   ```bash
   docker compose up -d
   ```

3. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -e ".[dev]"
   ```

4. Run the API:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

5. Visit http://localhost:8000/docs for interactive API docs.

## Project Structure

- `app/api/` — FastAPI routers (upload, query, feedback, jobs, health)
- `app/core/` — Database models, schemas, settings
- `app/ingestion/` — Parse, chunk, embed, index pipeline
- `app/retrieval/` — Hybrid search, re-rank, retrieval orchestrator
- `app/generation/` — Prompt builder, LLM client, streaming handler
- `app/evaluation/` — RAGAS and LLM-as-judge evaluation
- `app/utils/` — Shared utilities (file storage, logging, validators)
- `tests/` — Pytest suite
- `notebooks/` — Exploration and analysis notebooks
- `scripts/` — CLI helpers for seeding, evaluation, benchmarking

## Component Build Order

1. Core models + database
2. Ingestion pipeline (parser → chunker → embedder → indexer)
3. Query pipeline (hybrid search → re-rank → generate)
4. Streaming + citations
5. Evaluation + feedback loop
