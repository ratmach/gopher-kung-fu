from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_repo_env(*, override: bool = False) -> Path | None:
    """Load repo-root `.env` into os.environ (existing vars win unless override).

    Hugging Face Hub reads `HF_TOKEN` for authenticated downloads (higher rate limits).
    """
    path = ROOT / ".env"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not key or not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return path


load_repo_env()

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
