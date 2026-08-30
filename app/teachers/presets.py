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
    "qwen2.5-coder-3b": {
        "id": "qwen2.5-coder-3b",
        "label": "Qwen2.5-Coder 3B Instruct",
        "train_id": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        "merge_id": "unsloth/Qwen2.5-Coder-3B-Instruct",
        "default": False,
        "vram_hint": "~8 GB for QLoRA",
        "train_defaults": {
            "seq_len": 4096,
            "learning_rate": 1e-4,
        },
    },
    "qwen3-4b": {
        "id": "qwen3-4b",
        "label": "Qwen3 4B",
        "train_id": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "merge_id": "unsloth/Qwen3-4B",
        "default": False,
        "vram_hint": "~10 GB for QLoRA",
        "train_defaults": {
            "seq_len": 4096,
            "learning_rate": 1e-4,
        },
    },
    "qwen2.5-coder-7b": {
        "id": "qwen2.5-coder-7b",
        "label": "Qwen2.5-Coder 7B Instruct",
        "train_id": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "merge_id": "unsloth/Qwen2.5-Coder-7B-Instruct",
        "default": False,
        "vram_hint": "Prefer 12 GB+",
        "train_defaults": {
            "seq_len": 4096,
            "learning_rate": 1e-4,
        },
    },
    "deepseek-coder-v2": {
        "id": "deepseek-coder-v2",
        "label": "DeepSeek-Coder-V2 Lite Instruct",
        "train_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "merge_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "default": False,
        "vram_hint": "Prefer 16 GB+ (16B MoE, 2.4B active). Full 236B is not a local student.",
        "trust_remote_code": True,
        "train_defaults": {
            "seq_len": 4096,
            "learning_rate": 1e-4,
        },
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


def train_defaults_for(model_id: BaseModelId | str) -> dict[str, Any]:
    return dict(base_model_spec(model_id).get("train_defaults") or {})


def response_mask_parts(model_id: BaseModelId | str) -> dict[str, str]:
    key = str(model_id or "")
    if key.startswith("deepseek"):
        return {"instruction_part": "User:", "response_part": "Assistant:"}
    if key.startswith("qwen"):
        return {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        }
    return {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    }
