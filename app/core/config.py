from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://citerag:citerag@localhost:5432/citerag"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM / Ollama (OpenAI-compatible API)
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:11434/v1"
    openai_embedding_model: str = "nomic-embed-text:latest"
    openai_chat_model: str = "gemma4:31b-cloud"

    # Local fallback
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    # Retrieval
    default_top_k: int = 5
    reranker_top_k: int = 20
    enable_reranker: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings instance, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Legacy module-level alias — prefer get_settings() for lazy init
settings = get_settings()
