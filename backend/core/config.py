"""
Application configuration.

Settings are loaded from environment variables (and an optional local .env
file for development). No secrets are hard-coded here — see .env.example
for the variables this project expects.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Research Assistant backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General application settings
    app_name: str = "Research Assistant"
    environment: str = "development"
    debug: bool = True

    # API server settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database settings (PostgreSQL, per ARCHITECTURE.md §6)
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/research_assistant"

    # Filesystem locations for document data (see ARCHITECTURE.md §10)
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    vector_stores_dir: str = "data/vector_stores"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()