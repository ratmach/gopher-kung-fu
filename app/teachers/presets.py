from __future__ import annotations

from typing import Any

from app.models import BaseModelId

TEACHER_PRESETS: list[dict[str, Any]] = [
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "notes": "Official DeepSeek OpenAI-compatible API.",
    },
    {
        "id": "kimi",
        "label": "Kimi / K2.5",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.5",
        "notes": "Moonshot / Kimi. Change the model name if your account uses a different id.",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "notes": "One key, many teacher models. Set model to any OpenRouter id. Distill can optionally use OpenRouter's cheaper Batch API.",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
        "notes": "Check the provider ToS before using outputs as training data.",
    },
    {
        "id": "custom",
        "label": "Custom (OpenAI-compatible)",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "",
        "notes": "Any OpenAI-compatible endpoint: vLLM, llama-server, local gateway.",
    },
]

BASE_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-1.7b": {
        "id": "qwen3-1.7b",
        "label": "Qwen3 1.7B",
        "train_id": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "merge_id": "unsloth/Qwen3-1.7B",
        "default": True,
        "vram_hint": "~8 GB for QLoRA",
    },
    "ministral-3b": {
        "id": "ministral-3b",
        "label": "Ministral 3B",
        "train_id": "unsloth/Ministral-3-3B-Instruct-2512",
        "merge_id": "unsloth/Ministral-3-3B-Instruct-2512",
        "default": False,
        "vram_hint": "Prefer 12 GB+. Language layers only (vision off).",
        "vision": True,
    },
}


def preset_by_id(preset_id: str) -> dict[str, Any]:
    for item in TEACHER_PRESETS:
        if item["id"] == preset_id:
            return item
    raise KeyError(f"unknown teacher preset: {preset_id}")


def base_model_spec(model_id: BaseModelId | str) -> dict[str, Any]:
    if model_id not in BASE_MODELS:
        raise KeyError(f"unknown base model: {model_id}")
    return BASE_MODELS[model_id]
