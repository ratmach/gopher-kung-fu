from __future__ import annotations

import json

from fastapi import APIRouter

from app.catalog import load_catalog
from app.paths import CARTRIDGES_DIR
from app.pipeline.gpu import probe_torch
from app.teachers.presets import BASE_MODELS, TEACHER_PRESETS

router = APIRouter(tags=["meta"])


@router.get("/catalog")
def catalog() -> dict:
    return load_catalog()


@router.get("/teachers")
def teachers() -> dict:
    return {"presets": TEACHER_PRESETS}


@router.get("/base-models")
def base_models() -> dict:
    return {"models": list(BASE_MODELS.values())}


@router.get("/runtime")
def runtime() -> dict:
    return probe_torch()


@router.get("/cartridges")
def cartridges() -> dict:
    items = []
    if CARTRIDGES_DIR.exists():
        for folder in sorted(CARTRIDGES_DIR.iterdir()):
            card = folder / "card.json"
            if not card.is_file():
                continue
            data = json.loads(card.read_text(encoding="utf-8"))
            gguf = folder / data.get("gguf", "")
            items.append({**data, "ready": gguf.is_file(), "path": str(folder)})
    return {"cartridges": items}
