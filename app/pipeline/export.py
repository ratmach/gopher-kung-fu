from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.models import Project
from app.paths import cartridge_dir, project_dir
from app.pipeline.gpu import spawn_python_worker, unsloth_worker_env
from app.teachers.presets import base_model_spec


def latest_run_id(slug: str) -> str | None:
    runs = project_dir(slug) / "runs"
    if not runs.exists():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir() and (p / "merged").exists()]
    if not dirs:
        dirs = [p for p in runs.iterdir() if p.is_dir() and (p / "adapter").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def write_card(project: Project, out_dir: Path, gguf_name: str) -> Path:
    spec = base_model_spec(project.base_model)
    forbidden = list(project.forbidden_imports or [])
    if "C" not in forbidden:
        forbidden.append("C")
    card = {
        "id": project.slug,
        "name": project.name,
        "base_model": project.base_model,
        "base_hf": spec["merge_id"],
        "topics": [topic.model_dump() for topic in project.topics],
        "description": ", ".join(topic.label for topic in project.topics) or project.name,
        "gguf": gguf_name,
        "quant": "Q4_K_M",
        "allowed_imports": list(project.allowed_imports or []),
        "forbidden_imports": forbidden,
    }
    path = out_dir / "card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path


def spawn_export(project: Project, run_id: str) -> subprocess.Popen:
    spec = base_model_spec(project.base_model)
    dest = cartridge_dir(project.slug)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "slug": project.slug,
        "name": project.name,
        "merged_dir": str(project_dir(project.slug) / "runs" / run_id / "merged"),
        "adapter_dir": str(project_dir(project.slug) / "runs" / run_id / "adapter"),
        "merge_id": spec["merge_id"],
        "out_dir": str(dest),
        "gguf_name": f"{project.slug}.Q4_K_M.gguf",
        "quants": ["Q4_K_M", "Q5_K_M"],
    }
    config = dest / "export.json"
    config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return spawn_python_worker(
        ["-m", "app.pipeline.export_worker", "--config", str(config)],
        unsloth_worker_env(),
        cwd=str(Path(__file__).resolve().parents[2]),
    )
