"""AgentService — unified LLM factory with per-agent model selection.

Supports:
    - Local vLLM (OpenAI-compatible API, no auth needed)
    - DeepSeek API (cloud, requires API key)
    - Any other OpenAI-compatible provider

Usage:
    from ppg_agent.llm_service import AgentService

    service = AgentService()
    llm = service.get_llm("supervisor")         # DeepSeek for reasoning
    llm = service.get_llm("signal_processor")   # local Qwen for tools
    llm = service.get_llm()                     # default (qwen_local)
"""

from __future__ import annotations

import os
from pathlib import Path

# Load <repo root>/.env (DEEPSEEK_API_KEY etc.) so the API key works no
# matter how the process is started (uvicorn, tests, scripts).  The repo
# root is two levels up from this module.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # noqa: BLE001 — dotenv is optional
    pass
from typing import Any

import yaml

from ppg_agent.config import MODEL_REGISTRY


class AgentService:
    """Centralized LLM service for all agents.

    Loads a model registry YAML and provides get_llm(key) for each agent.
    API keys are read from environment variables (never hardcoded).
    """

    def __init__(self, registry_path: str | Path | None = None):
        self.registry_path = Path(registry_path or MODEL_REGISTRY)
        self.registry = self._load_registry()
        self._instances: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self, key: str | None = None, **overrides) -> Any:
        """Get or create an LLM instance.

        Args:
            key: Model key in the registry.  If None, uses default_llm.
            **overrides: Override any registry field (base_url, model, temperature, etc.)

        Returns:
            A ChatOpenAI instance.

        Raises:
            ValueError: If the key is not in the registry.
            ImportError: If langchain_openai is not installed.
            ConnectionError: If the API key is not set (for cloud providers).
        """
        cache_key = key or self.registry.get("default_llm", "qwen_local")

        if cache_key not in self._instances:
            config = self._get_model_config(cache_key)
            merged = {**config, **{k: v for k, v in overrides.items() if v is not None}}
            self._instances[cache_key] = self._build_llm(merged)

        return self._instances[cache_key]

    def list_models(self) -> dict[str, str]:
        """Return {key: description} for all registered models."""
        models = self.registry.get("models", {})
        return {
            key: cfg.get("description", "")
            for key, cfg in models.items()
        }

    def test_connection(self, key: str | None = None) -> bool:
        """Quick connectivity test for one model."""
        try:
            llm = self.get_llm(key)
            response = llm.invoke("Hello")
            print(f"  [{key or 'default'}] OK — {response.content[:60].strip()}...")
            return True
        except Exception as e:
            print(f"  [{key or 'default'}] FAILED — {e}")
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.registry_path}\n"
                f"Copy models/model_registry.example.yaml to models/model_registry.yaml"
            )
        with self.registry_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Model registry must be a mapping: {self.registry_path}")
        return data

    def _get_model_config(self, key: str) -> dict[str, Any]:
        models = self.registry.get("models", {})
        if key not in models:
            available = ", ".join(models.keys())
            raise ValueError(
                f"Unknown model key '{key}'. Available: {available}"
            )
        return dict(models[key])

    def _build_llm(self, config: dict[str, Any]) -> Any:
        provider = config.get("provider", "openai_compatible")

        if provider == "openai_compatible":
            return self._build_openai_compatible(config)

        raise ValueError(f"Unsupported provider: {provider}")

    def _build_openai_compatible(self, config: dict[str, Any]) -> Any:
        from langchain_openai import ChatOpenAI

        base_url = config.get("base_url", "http://localhost:8000/v1")
        model = config.get("model", "qwen2.5-14b-instruct")
        temperature = float(config.get("temperature", 0.0))
        max_tokens = int(config.get("max_tokens", 4096))

        # Resolve API key
        api_key_env = config.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not api_key:
            # Fallback: check common env vars
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Local vLLM doesn't need a real key
            api_key = "sk-placeholder"

        return ChatOpenAI(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

_service: AgentService | None = None


def get_service() -> AgentService:
    """Get the global AgentService singleton."""
    global _service
    if _service is None:
        _service = AgentService()
    return _service


def get_llm(key: str | None = None, **overrides) -> Any:
    """Convenience: get an LLM from the global service."""
    return get_service().get_llm(key, **overrides)
