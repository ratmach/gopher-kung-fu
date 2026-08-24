from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.models import Project
from app.paths import project_dir
from app.pipeline.gpu import require_cuda, spawn_python_worker, unsloth_worker_env
from app.teachers.presets import base_model_spec


def run_dir(slug: str, run_id: str) -> Path:
    return project_dir(slug) / "runs" / run_id


def write_train_config(project: Project, run_id: str) -> Path:
    spec = base_model_spec(project.base_model)
    folder = run_dir(project.slug, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "slug": project.slug,
        "run_id": run_id,
        "base_model": project.base_model,
        "train_id": spec["train_id"],
        "merge_id": spec["merge_id"],
        "vision": bool(spec.get("vision")),
        "adapter_dir": str(folder / "adapter"),
        "merged_dir": str(folder / "merged"),
        "train_jsonl": str(project_dir(project.slug) / "synthetic" / "train.jsonl"),
        "eval_jsonl": str(project_dir(project.slug) / "synthetic" / "eval.jsonl"),
        "lora_r": project.train.lora_r,
        "lora_alpha": project.train.lora_alpha,
        "epochs": project.train.epochs,
        "seq_len": project.train.seq_len,
        "batch_size": project.train.batch_size,
        "grad_accum": project.train.grad_accum,
        "learning_rate": project.train.learning_rate,
    }
    path = folder / "train.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def spawn_train(config_path: Path) -> subprocess.Popen:
    gpu = require_cuda()
    env = unsloth_worker_env()
    if gpu.get("device_name"):
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return spawn_python_worker(
        ["-m", "app.pipeline.train_worker", "--config", str(config_path)],
        env,
    )
