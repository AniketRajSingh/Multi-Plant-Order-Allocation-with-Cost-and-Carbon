"""Central configuration loader.

Reads model and environment settings dynamically from:
1. `config.json` (or path in CONFIG_PATH / CONFIG_FILE)
2. `.env` file in project root
3. System environment variables

No models or endpoints are hardcoded in application logic.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("config")

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _load_env_file(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        logger.warning("Failed to parse .env file: %s", exc)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json, .env, and environment variables."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    root = Path(__file__).resolve().parents[1]
    _load_env_file(root)

    cfg: Dict[str, Any] = {}

    config_path = os.getenv("CONFIG_PATH") or os.getenv("CONFIG_FILE")
    target_json = Path(config_path) if config_path else root / "config.json"

    if target_json.exists():
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", target_json, exc)

    _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


def get_llm_model() -> str:
    """Retrieve the LLM model name strictly from config or environment."""
    cfg = load_config()
    model = (
        cfg.get("llm_model")
        or cfg.get("model")
        or os.getenv("LLM_MODEL")
        or os.getenv("OLLAMA_MODEL")
    )
    if not model:
        # Fall back to empty string or mock indicator if no model is declared
        return ""
    return str(model).strip()


def get_llm_provider() -> str:
    """Retrieve LLM provider (e.g. 'ollama', 'openai', etc.)."""
    cfg = load_config()
    return str(
        cfg.get("llm_provider")
        or cfg.get("provider")
        or os.getenv("LLM_PROVIDER")
        or "ollama"
    ).lower()


def get_ollama_base_url() -> str:
    """Retrieve Ollama base URL strictly from config or environment."""
    cfg = load_config()
    url = (
        cfg.get("ollama_base_url")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )
    return str(url).rstrip("/")


def get_mock_llm() -> bool:
    """Check if mock mode is requested."""
    cfg = load_config()
    if "mock_llm" in cfg:
        val = cfg["mock_llm"]
        return str(val).lower() == "true" if isinstance(val, str) else bool(val)
    return os.getenv("MOCK_LLM", "false").lower() == "true"


def get_db_path() -> str:
    """Retrieve SQLite database path."""
    cfg = load_config()
    return str(
        cfg.get("db_path")
        or os.getenv("DB_PATH")
        or os.getenv("SQLITE_PATH")
        or "data/allocation.db"
    )


def get_planning_start() -> str:
    """Retrieve planning start date string."""
    cfg = load_config()
    return str(
        cfg.get("planning_start")
        or os.getenv("PLANNING_START")
        or "2026-10-12"
    )
