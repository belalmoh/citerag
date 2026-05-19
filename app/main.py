from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, upload, query, feedback, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here (DB init, connections, etc.)
    yield
    # Shutdown logic goes here (close connections, etc.)


app = FastAPI(
    title="CiteRAG",
    description="Production-Grade RAG Document System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.get("/")
async def root():
    return {"message": "CiteRAG is running", "version": "0.1.0"}
