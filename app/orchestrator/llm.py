"""LLM Client integration for local Ollama instance.

Connects to Ollama instance (default: http://192.168.3.173:11434)
using OpenAI-compatible endpoints with support for reasoning models (e.g. jev-9b:latest).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI


logger = logging.getLogger("orchestrator_llm")

# Default Ollama configuration
DEFAULT_OLLAMA_BASE_URL = "http://192.168.3.173:11434"
DEFAULT_OLLAMA_MODEL = "jev-9b:latest"


def load_env_file() -> None:
    """Load configuration from local uncommitted .env file if present."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        logger.warning(f"Could not parse .env file: {e}")


# Load local env on import
load_env_file()


def get_ollama_base_url() -> str:
    """Retrieve configured Ollama base URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def get_ollama_model() -> str:
    """Retrieve configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_configured_llm(temperature: float = 0.0, timeout: float = 60.0) -> ChatOpenAI:
    """Instantiate ChatOpenAI client bound to Ollama instance."""
    base_url = f"{get_ollama_base_url()}/v1"
    model = get_ollama_model()

    return ChatOpenAI(
        base_url=base_url,
        api_key="ollama",
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


def clean_llm_response(raw_text: str) -> str:
    """Strip reasoning tokens (<think>...</think>) from Qwen/DeepSeek/Jev models."""
    if not raw_text:
        return ""
    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    # Strip leading/trailing whitespaces and markdown escapes
    return cleaned.strip()


def invoke_brief_llm(prompt: str) -> Tuple[str, Dict[str, int]]:
    """Invoke the configured Ollama LLM and return cleaned content with token usage.

    Returns:
        Tuple of (cleaned_text, token_usage_dict)
    """
    llm = get_configured_llm()
    resp = llm.invoke(prompt)

    raw_content = resp.content if hasattr(resp, "content") else str(resp)
    cleaned = clean_llm_response(raw_content)

    meta = getattr(resp, "response_metadata", {}) or {}
    tok_usage = meta.get("token_usage", {}) or {}

    prompt_tokens = int(tok_usage.get("prompt_tokens", len(prompt) // 4))
    completion_tokens = int(tok_usage.get("completion_tokens", len(cleaned) // 4))
    total_tokens = int(tok_usage.get("total_tokens", prompt_tokens + completion_tokens))

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return cleaned, usage
