import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import MODEL_REGISTRY


def load_model_registry(registry_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the local model registry if it exists.

    The private registry is intentionally optional. Demos can still use
    environment variables or explicit arguments on machines without a registry.
    """
    path = Path(registry_path or MODEL_REGISTRY)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Model registry must be a mapping: {path}")
    return data


def get_llm(base_url: str = None, model: str = None, temperature: float = 0, registry_key: str = None):
    """获取 LLM 实例

    Args:
        base_url: vLLM API 地址，如 "http://localhost:8000/v1"
        model: 模型名称，如 "qwen2.5-14b-instruct"
        temperature: 温度参数，越低越稳定
        registry_key: models/model_registry.yaml 中的模型键名

    Returns:
        ChatOpenAI: LLM 实例
    """
    registry = load_model_registry()
    default_key = registry.get("default_llm")
    selected_key = registry_key or default_key
    selected_model = (registry.get("models") or {}).get(selected_key, {}) if selected_key else {}

    if base_url is None:
        base_url = os.environ.get(
            'VLLM_API_URL',
            selected_model.get("base_url", "http://localhost:8000/v1"),
        )

    if model is None:
        model = os.environ.get(
            'LLM_MODEL',
            selected_model.get("model", "qwen2.5-14b-instruct"),
        )

    api_key_env = selected_model.get("api_key_env", "PPG_LLM_API_KEY")
    api_key = os.environ.get(api_key_env, os.environ.get("OPENAI_API_KEY", "sk-placeholder"))

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain_openai is required for LLM calls. Install project agent "
            "dependencies before running interactive demos."
        ) from exc

    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    return llm


def test_llm_connection(llm):
    """测试 LLM 连接是否正常

    Args:
        llm: LLM 实例

    Returns:
        bool: 是否连接成功
    """
    try:
        response = llm.invoke("Hello, can you hear me?")
        print(f"LLM 连接成功！响应: {response.content[:50]}...")
        return True
    except Exception as e:
        print(f"LLM 连接失败: {e}")
        print("请确保 vLLM 服务已启动")
        return False
