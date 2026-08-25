from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CUSTOM_SLM_DATA", ROOT / "data")).resolve()
PROJECTS_DIR = DATA_DIR / "projects"
LIBRARY_DIR = DATA_DIR / "library"
CARTRIDGES_DIR = Path(os.environ.get("CUSTOM_SLM_CARTRIDGES", ROOT / "cartridges")).resolve()
SECRETS_PATH = DATA_DIR / "secrets.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
UNSLOTH_CACHE = DATA_DIR / "unsloth_compiled_cache"
LLAMA_CPP_TOOLS = DATA_DIR / "llama_cpp"
CATALOG_PATH = Path(__file__).resolve().parent / "catalog" / "topics.yaml"
WEB_DIST = ROOT / "web" / "dist"


def ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    CARTRIDGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UNSLOTH_CACHE.mkdir(parents=True, exist_ok=True)
    LLAMA_CPP_TOOLS.mkdir(parents=True, exist_ok=True)


def project_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug


def cartridge_dir(slug: str) -> Path:
    return CARTRIDGES_DIR / slug
