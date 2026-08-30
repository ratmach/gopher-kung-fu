from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 400_000
MAX_REPAIR_FILE_CHARS = 24_000
HEAD = re.compile(
    r"^###\s+`?(?P<path>[A-Za-z0-9_./\\-]+\.go)`?\s*$",
    re.MULTILINE,
)
GO_FENCE = re.compile(r"```go[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n(.*?)```", re.DOTALL)


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


def parse_wanted_files(files: str) -> list[str]:
    wanted: list[str] = []
    seen: set[str] = set()
    for line in (files or "").splitlines():
        raw = line.strip().strip("`")
        if not raw or raw.startswith("#"):
            continue
        rel = normalize_rel(raw)
        if rel not in seen:
            seen.add(rel)
            wanted.append(rel)
    return wanted


def _looks_like_go(body: str) -> bool:
    stripped = body.lstrip()
    return stripped.startswith("package ") or stripped.startswith("func ") or "\nfunc " in body


def _body_from_section(section: str) -> str | None:
    go = list(GO_FENCE.finditer(section))
    if go:
        return go[-1].group(1)
    for match in reversed(list(ANY_FENCE.finditer(section))):
        body = match.group(1)
        if _looks_like_go(body):
            return body
    return None


def parse_specialist_files(text: str) -> list[ParsedFile]:
    source = text or ""
    files: list[ParsedFile] = []
    seen: set[str] = set()
    headings = list(HEAD.finditer(source))
    for index, match in enumerate(headings):
        rel = normalize_rel(match.group("path"))
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(source)
        body_raw = _body_from_section(source[start:end])
        if body_raw is None:
            raise ApplyError(f"no code fence after ### {rel}")
        body = body_raw.replace("\r\n", "\n")
        if not body.endswith("\n"):
            body += "\n"
        if len(body.encode("utf-8")) > MAX_FILE_BYTES:
            raise ApplyError(f"{rel} exceeds {MAX_FILE_BYTES} bytes")
        if rel in seen:
            files = [item for item in files if item.rel != rel]
        seen.add(rel)
        files.append(ParsedFile(rel=rel, body=body))
    return files


def map_parsed_to_wanted(
    parsed: list[ParsedFile],
    wanted: list[str],
) -> tuple[list[ParsedFile], list[str]]:
    """Keep requested paths only; remap a unique basename (merge.go → internal/merge/merge.go)."""
    if not wanted:
        return list(parsed), []
    by_rel = {item.rel: item for item in parsed}
    used: set[str] = set()
    mapped: list[ParsedFile] = []
    for target in wanted:
        if target in by_rel:
            mapped.append(ParsedFile(rel=target, body=by_rel[target].body))
            used.add(target)
            continue
        base = Path(target).name
        wanted_same = [path for path in wanted if Path(path).name == base]
        candidates = [item for item in parsed if Path(item.rel).name == base and item.rel not in used]
        if len(wanted_same) == 1 and len(candidates) == 1:
            mapped.append(ParsedFile(rel=target, body=candidates[0].body))
            used.add(candidates[0].rel)
    extras = [item.rel for item in parsed if item.rel not in used]
    return mapped, extras


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


def history_hash_path(root: Path, rel: str) -> Path:
    """`.gopher/history/<rel>_history.hash` — one sha256 per distinct on-disk version."""
    rel_n = normalize_rel(rel)
    return root / ".gopher" / "history" / (rel_n + "_history.hash")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_history_digest(path: Path) -> str:
    if not path.is_file():
        return ""
    last = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        last = raw.split()[-1]
    return last


def record_write_hashes(root: Path, rels: list[str], *, attempt: int) -> list[str]:
    """Append sha256 when the written file changed vs the last history line."""
    recorded: list[str] = []
    for rel in rels:
        src = root / rel
        if not src.is_file():
            continue
        digest = file_sha256(src)
        dest = history_hash_path(root, rel)
        if digest == _last_history_digest(dest):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{attempt} {digest}\n")
        recorded.append(rel)
    return recorded


def previous_attempt_blob(root: Path, rels: list[str]) -> str:
    parts: list[str] = []
    for rel in rels:
        path = root / rel
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8")
        if len(body) > MAX_REPAIR_FILE_CHARS:
            body = body[:MAX_REPAIR_FILE_CHARS] + "\n// ... truncated\n"
        parts.append(f"### {rel}\n```go\n{body.rstrip()}\n```")
    return "\n\n".join(parts)


def format_apply_summary(
    root: Path,
    written: list[str],
    *,
    skipped: list[str] | None = None,
    skipped_extra: list[str] | None = None,
    missing: list[str] | None = None,
    test_line: str = "",
) -> str:
    lines = [f"Wrote {len(written)} file(s) under {root}"]
    for rel in written:
        path = root / rel
        size = path.stat().st_size if path.is_file() else 0
        lines.append(f"- {rel} ({size} bytes)")
    for rel in skipped or []:
        lines.append(f"- skipped (frozen test) {rel}")
    for rel in skipped_extra or []:
        lines.append(f"- skipped (not in files=) {rel}")
    for rel in missing or []:
        lines.append(f"- missing requested {rel}")
    if test_line:
        lines.append(test_line)
    return "\n".join(lines)
