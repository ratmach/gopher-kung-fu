from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.paths import CARTRIDGES_DIR

_IMPORT_BLOCK = re.compile(
    r"(?m)^\s*import\s*(?:\(\s*([\s\S]*?)\)|\"([^\"]+)\")",
)
_QUOTED = re.compile(r'"([^"]+)"')
_CGO = re.compile(r"(?m)^\s*#cgo\b")
_NEG_CONSTRAINT = re.compile(
    r"\b(never|don't|do not|must not|cannot|forbidden|forbid|without using|"
    r"not use|no import|never import|do not import|don't import)\b",
    re.I,
)
_IMPORT_PATH = re.compile(r"\b([a-zA-Z][\w.-]+(?:/[\w.-]+)+)\b")


def is_stdlib(path: str) -> bool:
    first = (path or "").split("/")[0]
    return bool(first) and first != "C" and "." not in first


def go_imports(source: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for block, single in _IMPORT_BLOCK.findall(source or ""):
        paths = [single] if single else _QUOTED.findall(block)
        for path in paths:
            if path and path not in seen:
                seen.add(path)
                found.append(path)
    return found


def _allowed(path: str, allowed: list[str]) -> bool:
    for item in allowed:
        prefix = item.rstrip("/")
        if not prefix:
            continue
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def import_violations(
    source: str,
    *,
    allowed_imports: list[str] | None = None,
    forbidden_imports: list[str] | None = None,
) -> list[str]:
    allowed = list(allowed_imports or [])
    forbidden = list(forbidden_imports or [])
    hits: list[str] = []
    seen: set[str] = set()
    if _CGO.search(source or "") and "C" not in seen:
        if "C" in forbidden or not allowed:
            hits.append("C")
            seen.add("C")
    for path in go_imports(source):
        if path in seen:
            continue
        if path in forbidden or path == "C":
            hits.append(path)
            seen.add(path)
            continue
        if is_stdlib(path):
            continue
        if _allowed(path, allowed):
            continue
        hits.append(path)
        seen.add(path)
    return hits


def load_cartridge_card(model_id: str) -> dict[str, Any]:
    wanted = (model_id or "").strip()
    if not wanted or not CARTRIDGES_DIR.is_dir():
        return {}
    for folder in sorted(CARTRIDGES_DIR.iterdir()):
        path = folder / "card.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and str(data.get("id") or "") == wanted:
            return data
    return {}


def forbidden_from_constraints(text: str) -> list[str]:
    """Pull import paths out of 'never import encoding/csv' style constraint lines."""
    raw = text or ""
    if not raw.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()
    chunks = re.split(r"[\n;,]+", raw)
    chunks.append(raw)
    for chunk in chunks:
        if not _NEG_CONSTRAINT.search(chunk):
            continue
        for path in _IMPORT_PATH.findall(chunk):
            if path.startswith("http"):
                continue
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def resolve_allowlists(
    model_id: str | None,
    *,
    allowed_imports: list[str] | None = None,
    forbidden_imports: list[str] | None = None,
    constraints: str = "",
) -> tuple[list[str], list[str]]:
    card_allowed, card_forbidden = card_allowlists(load_cartridge_card((model_id or "").strip()))
    allowed = card_allowed if allowed_imports is None else list(allowed_imports)
    forbidden = card_forbidden if forbidden_imports is None else list(forbidden_imports)
    for path in forbidden_from_constraints(constraints):
        if path not in forbidden:
            forbidden.append(path)
    return allowed, forbidden


def card_allowlists(card: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    data = card or {}
    allowed = [str(item).strip() for item in (data.get("allowed_imports") or []) if str(item).strip()]
    forbidden = [str(item).strip() for item in (data.get("forbidden_imports") or ["C"]) if str(item).strip()]
    if "C" not in forbidden:
        forbidden.append("C")
    return allowed, forbidden


def check_written_imports(
    root: Path,
    rels: list[str],
    *,
    allowed_imports: list[str] | None,
    forbidden_imports: list[str] | None,
) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for rel in rels:
        path = root / rel
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for item in import_violations(
            source,
            allowed_imports=allowed_imports,
            forbidden_imports=forbidden_imports,
        ):
            if item not in seen:
                seen.add(item)
                hits.append(item)
    return hits
