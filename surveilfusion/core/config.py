from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SurveilFusion"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    public_base_url: AnyHttpUrl | None = None
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SURVEILFUSION_API_KEY", "API_KEY"),
    )
    data_dir: Path = Path("data")
    cameras_file: Path = Path("config/cameras.example.yml")
    database_url: str = "sqlite+aiosqlite:///data/surveilfusion.db"

    llm_provider: Literal["disabled", "openai", "ollama"] = "disabled"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2-vision"

    mqtt_url: str | None = "mqtt://mqtt:1883"
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    notification_webhooks: str = ""
    enable_notifications: bool = False

    fire_model_path: Path = Path("models/fire-yolo.pt")
    person_model_path: Path | None = None
    default_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    event_cooldown_seconds: int = Field(default=20, ge=0)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
