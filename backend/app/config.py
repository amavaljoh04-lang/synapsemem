"""Application settings, loaded from env + sensible defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend service."""

    model_config = SettingsConfigDict(
        env_prefix="SYNAPSEMEM_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("/app/data"),
        description="Directory where SQLite and derived indices live.",
    )
    database_url: str = Field(
        default="",
        description="SQLAlchemy URL. Empty → derived from data_dir/synapse.db.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Default Ollama server for embeddings + extraction.",
    )
    embeddings_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model used to embed text episodes + symbols.",
    )
    extractor_model: str = Field(
        default="qwen2.5-coder:32b",
        description="Ollama model used by the LLM extractor / chat. Single model by design.",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins for the embedded frontend.",
    )

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.data_dir}/synapse.db"

    def ensure_data_dir(self) -> None:
        """Create ``data_dir`` if it doesn't exist. Call at startup, not at import."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
