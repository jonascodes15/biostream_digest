"""Environment-driven configuration for BioStreamer.

Reads from the process environment, falling back to the local docker-compose
defaults so the platform runs out of the box in this Codespace. Application
secrets belong in .env (git-ignored); see .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: no dependency, does not overwrite real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PostgresSettings:
    host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))
    user: str = field(default_factory=lambda: _env("POSTGRES_USER", "data_engineer"))
    password: str = field(
        default_factory=lambda: _env("POSTGRES_PASSWORD", "zoomcamp_secret_pass")
    )
    database: str = field(default_factory=lambda: _env("POSTGRES_DB", "biostream_db"))
    schema: str = field(default_factory=lambda: _env("BIOSTREAM_SCHEMA", "bioprocess"))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class QdrantSettings:
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    collection: str = field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "bioprocess_knowledge")
    )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = field(default_factory=lambda: _env("MINIO_ROOT_USER", "admin"))
    secret_key: str = field(
        default_factory=lambda: _env("MINIO_ROOT_PASSWORD", "adminpassword")
    )
    bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET", "biostreamer-lake"))
    secure: bool = False


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None
    )
    # Defaults to the current flagship. Override in .env to trade cost for depth.
    model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("ANTHROPIC_MAX_TOKENS", 2048))

    @property
    def enabled(self) -> bool:
        """False means /chat degrades to retrieval-only rather than failing."""
        return bool(self.api_key)


@dataclass(frozen=True)
class SimulationSettings:
    seed: int = field(default_factory=lambda: _env_int("BIOSTREAM_SEED", 20240423))
    reactor_count: int = field(default_factory=lambda: _env_int("BIOSTREAM_REACTORS", 100))
    timeline_days: int = field(default_factory=lambda: _env_int("BIOSTREAM_DAYS", 30))
    start_date: str = field(default_factory=lambda: _env("BIOSTREAM_START_DATE", "2024-10-01"))


@dataclass(frozen=True)
class Settings:
    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    minio: MinioSettings = field(default_factory=MinioSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)

    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dim: int = 384
    api_base_url: str = field(
        default_factory=lambda: _env("BIOSTREAM_API_URL", "http://localhost:8000")
    )


def get_settings() -> Settings:
    return Settings()
