"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    anthropic_api_key: str = ""
    extraction_model: str = "openai/gpt-oss-120b"

    edgar_user_agent: str = ""
    database_url: str = "postgresql+psycopg://filinglens:filinglens@localhost:5432/filinglens"

    # Max characters of filing text sent for extraction (~4 chars per token).
    extraction_max_chars: int = 60_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
