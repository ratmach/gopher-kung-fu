from __future__ import annotations

import shutil
from pathlib import Path

from app.paths import cartridge_dir, project_dir


def format_bytes(n: int) -> str:
    value = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(n)} B"


def path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def remove_path(path: Path) -> int:
    """Delete a file or tree. Returns bytes that were on disk before the call."""
    if not path.exists():
        return 0
    freed = path_bytes(path)
    if path.is_file() or path.is_symlink():
        try:
            path.unlink()
        except OSError:
            return 0
        return freed
    shutil.rmtree(path, ignore_errors=True)
    return 0 if path.exists() else freed


def cleanup_after_train(run_folder: Path) -> tuple[list[str], int]:
    """Drop trainer checkpoints after a successful train. Keep adapter + merged for export."""
    return _remove_named(run_folder, [Path("adapter") / "trainer"])


def cleanup_after_export(slug: str, run_id: str) -> tuple[list[str], int]:
    """Drop merged HF weights, trainer leftovers, f16 temps, and older runs after GGUF is written."""
    removed: list[str] = []
    freed = 0
    run_folder = project_dir(slug) / "runs" / run_id
    extra, extra_bytes = _remove_named(
        run_folder,
        [Path("merged"), Path("adapter") / "trainer"],
    )
    removed.extend(extra)
    freed += extra_bytes

    dest = cartridge_dir(slug)
    if dest.is_dir():
        for leftover in list(dest.glob("*.f16.gguf")) + [dest / "_gguf_tmp"]:
            n = remove_path(leftover)
            if n:
                removed.append(str(leftover))
                freed += n

    runs = project_dir(slug) / "runs"
    if runs.is_dir():
        for child in runs.iterdir():
            if not child.is_dir() or child.name == run_id:
                continue
            n = remove_path(child)
            if n:
                removed.append(str(child))
                freed += n
    return removed, freed


def _remove_named(root: Path, rels: list[Path]) -> tuple[list[str], int]:
    removed: list[str] = []
    freed = 0
    for rel in rels:
        path = root / rel
        n = remove_path(path)
        if n:
            removed.append(str(path))
            freed += n
    return removed, freed
