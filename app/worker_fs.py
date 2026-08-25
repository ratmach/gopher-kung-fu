from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 400_000
HEAD = re.compile(
    r"^###\s+`?(?P<path>[A-Za-z0-9_./\\-]+\.go)`?\s*$",
    re.MULTILINE,
)
FENCE = re.compile(r"```(?:go)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class ApplyError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedFile:
    rel: str
    body: str


def normalize_rel(path: str) -> str:
    rel = (path or "").strip().strip("`").replace("\\", "/")
    if not rel.lower().endswith(".go"):
        raise ApplyError(f"refusing non-Go path: {path!r}")
    if rel.startswith("/") or re.match(r"^[A-Za-z]:/", rel):
        raise ApplyError(f"refusing absolute path: {path!r}")
    parts: list[str] = []
    for piece in rel.split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            raise ApplyError(f"refusing path escape: {path!r}")
        parts.append(piece)
    if not parts:
        raise ApplyError(f"refusing empty path: {path!r}")
    return "/".join(parts)


def workspace_root(workspace: str | None = None) -> Path:
    raw = (workspace or "").strip() or os.environ.get("GOPHER_WORKSPACE", "").strip()
    if not raw:
        raise ApplyError(
            "workspace is required (implement_go workspace= or GOPHER_WORKSPACE). "
            "Use the Go module root, not this farm repo."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ApplyError(f"workspace is not a directory: {root}")
    return root


def resolve_under(root: Path, rel: str) -> Path:
    rel_n = normalize_rel(rel)
    dest = (root / rel_n).resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ApplyError(f"refusing path outside workspace: {rel}") from exc
    return dest


def parse_specialist_files(text: str) -> list[ParsedFile]:
    source = text or ""
    files: list[ParsedFile] = []
    seen: set[str] = set()
    for match in HEAD.finditer(source):
        rel = normalize_rel(match.group("path"))
        rest = source[match.end() :]
        fence = FENCE.search(rest)
        if not fence:
            raise ApplyError(f"no code fence after ### {rel}")
        body = fence.group(1).replace("\r\n", "\n")
        if not body.endswith("\n"):
            body += "\n"
        if len(body.encode("utf-8")) > MAX_FILE_BYTES:
            raise ApplyError(f"{rel} exceeds {MAX_FILE_BYTES} bytes")
        if rel in seen:
            files = [item for item in files if item.rel != rel]
        seen.add(rel)
        files.append(ParsedFile(rel=rel, body=body))
    return files


def apply_files(
    root: Path,
    files: list[ParsedFile],
    *,
    freeze_tests: bool = False,
) -> tuple[list[str], list[str]]:
    written: list[str] = []
    skipped: list[str] = []
    for item in files:
        dest = resolve_under(root, item.rel)
        if freeze_tests and dest.name.endswith("_test.go"):
            skipped.append(item.rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(item.body, encoding="utf-8", newline="\n")
        written.append(item.rel)
    return written, skipped


def format_apply_summary(
    root: Path,
    written: list[str],
    *,
    skipped: list[str] | None = None,
    test_line: str = "",
) -> str:
    lines = [f"Wrote {len(written)} file(s) under {root}"]
    for rel in written:
        path = root / rel
        size = path.stat().st_size if path.is_file() else 0
        lines.append(f"- {rel} ({size} bytes)")
    for rel in skipped or []:
        lines.append(f"- skipped (frozen test) {rel}")
    if test_line:
        lines.append(test_line)
    return "\n".join(lines)
