from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.worker_gopls import missing_method_from_message, type_from_missing_message, unused_name_from_message

MAX_MEMBERS = 15
GODOC_TIMEOUT = 12.0
IMPORTS_TIMEOUT = 20.0

_NO_MEMBER = re.compile(
    r"(?P<expr>[\w.]+)\.(?P<want>\w+)\s+undefined\s+\(type\s+(?P<typ>.+?)\s+"
    r"has no field or method\s+(?P=want)\)"
)
_UNDEFINED = re.compile(r"undefined:\s+(?P<name>[\w.]+)")
_IMPORT_LINE = re.compile(
    r'^\s*(?:(\w+|\.|_)\s+)?"([^"]+)"\s*$',
)
_IMPORT_BLOCK = re.compile(r"(?m)^\s*import\s*\(\s*([\s\S]*?)\)")
_IMPORT_SINGLE = re.compile(r'(?m)^\s*import\s+(?:(\w+|\.|_)\s+)?"([^"]+)"')
_METHOD = re.compile(
    r"^func\s+\((?P<recv>[^)]+)\)\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_TYPE_STRUCT = re.compile(
    r"^type\s+(?P<name>\w+)\s+struct\s*\{(?P<body>[^{}]*)\}",
    re.MULTILINE,
)
_DOC_METHOD = re.compile(
    r"^func \([^)]+\) ([A-Za-z_]\w*)\(",
    re.MULTILINE,
)
_DOC_FUNC = re.compile(
    r"^func ([A-Za-z_]\w*)\(",
    re.MULTILINE,
)
_EXPORTED_FIELD = re.compile(r"^\s*([A-Z]\w*)\s+\S", re.MULTILINE)
_IFACE_METHOD = re.compile(r"(?m)^\s+([A-Z][A-Za-z0-9]*)\(")
_VERSION_SUFFIX = re.compile(r"^v\d+$")
_SELECTOR_FIX_RATIO = 0.5
_UNUSED_GC = re.compile(
    r"(?P<rel>[^\s:]+?\.go):(?P<line>\d+):\d+:\s*declared and not used:\s+(?P<name>\w+)",
    re.IGNORECASE,
)
UNUSED_ACCUM_HINT = (
    "declared and not used: use the named local (append, increment, return it). "
    "`_` only for an unused range index (`for _, v := range`), never `_ = buf`."
)


@dataclass(frozen=True)
class TypeMiss:
    expr: str
    want: str
    type_str: str
    kind: str = "member"  # member | pkg


def parse_type_errors(output: str) -> list[TypeMiss]:
    found: list[TypeMiss] = []
    seen: set[tuple[str, str, str]] = set()
    text = output or ""
    for match in _NO_MEMBER.finditer(text):
        miss = TypeMiss(
            expr=match.group("expr"),
            want=match.group("want"),
            type_str=match.group("typ").strip(),
            kind="member",
        )
        key = (miss.kind, miss.type_str, miss.want)
        if key not in seen:
            seen.add(key)
            found.append(miss)
    for match in _UNDEFINED.finditer(text):
        name = match.group("name")
        if "." not in name:
            continue
        pkg, want = name.rsplit(".", 1)
        if not pkg or not want:
            continue
        miss = TypeMiss(expr=pkg, want=want, type_str=pkg, kind="pkg")
        key = (miss.kind, miss.type_str, miss.want)
        if key not in seen:
            seen.add(key)
            found.append(miss)
    return found


def import_aliases(source: str) -> dict[str, str]:
    """Map selector name -> import path (skip dot and blank imports)."""
    aliases: dict[str, str] = {}
    for block in _IMPORT_BLOCK.findall(source or ""):
        for line in block.splitlines():
            match = _IMPORT_LINE.match(line.strip())
            if not match:
                continue
            name, path = match.group(1), match.group(2)
            if name in {".", "_"}:
                continue
            aliases[name or _default_alias(path)] = path
    for name, path in _IMPORT_SINGLE.findall(source or ""):
        if name in {".", "_"}:
            continue
        aliases[name or _default_alias(path)] = path
    return aliases


def _default_alias(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) >= 2 and _VERSION_SUFFIX.match(parts[-1]):
        return parts[-2]
    return parts[-1] if parts else path


def _strip_type(type_str: str) -> str:
    raw = (type_str or "").strip()
    while raw.startswith("*"):
        raw = raw[1:].strip()
    return raw


def _recv_type(recv: str) -> str:
    token = (recv or "").replace("*", " ").split()
    if not token:
        return ""
    name = token[-1]
    return name.rsplit(".", 1)[-1]


def members_from_source(sources: list[str], type_name: str) -> list[str]:
    want = _strip_type(type_name).rsplit(".", 1)[-1]
    if not want or want in {"map", "chan", "func", "interface", "struct"}:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for match in _METHOD.finditer(source or ""):
            if _recv_type(match.group("recv")) != want:
                continue
            name = match.group("name")
            if name[:1].isupper() and name not in seen:
                seen.add(name)
                names.append(name)
        for match in _TYPE_STRUCT.finditer(source or ""):
            if match.group("name") != want:
                continue
            for field in _EXPORTED_FIELD.findall(match.group("body") or ""):
                if field not in seen:
                    seen.add(field)
                    names.append(field)
    return names


def members_from_godoc(text: str, *, funcs: bool = False) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    patterns = [_DOC_FUNC, _DOC_METHOD] if funcs else [_DOC_METHOD]
    for pattern in patterns:
        for name in pattern.findall(text or ""):
            if name[:1].isupper() and name not in seen:
                seen.add(name)
                names.append(name)
    if not funcs:
        for name in _EXPORTED_FIELD.findall(text or ""):
            if name not in seen:
                seen.add(name)
                names.append(name)
        for name in _IFACE_METHOD.findall(text or ""):
            if name[:1].isupper() and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def pick_selector_fix(want: str, names: list[str]) -> str | None:
    """Replace an invented selector when gopls has one close, unique match."""
    if not want or not names:
        return None
    if want in names:
        return None
    ranked = rank_members(want, names)
    if not ranked:
        return None
    close = ranked[0]
    target = want.lower()
    best = close.lower()
    prefix = bool(target) and (target.startswith(best) or best.startswith(target))
    ratio = SequenceMatcher(None, target, best).ratio() if target else 0.0
    second = SequenceMatcher(None, target, ranked[1].lower()).ratio() if len(ranked) > 1 else 0.0
    if prefix and (len(ranked) == 1 or ratio >= second + 0.08):
        return close
    if ratio >= _SELECTOR_FIX_RATIO and (len(ranked) == 1 or second < 0.45):
        return close
    return None


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    start = 0
    while True:
        idx = text.find("\n", start)
        if idx < 0:
            break
        offsets.append(idx + 1)
        start = idx + 1
    return offsets


def _span(text: str, line: int, character: int, end_line: int, end_character: int) -> tuple[int, int]:
    offsets = _line_offsets(text)
    if line < 0 or line >= len(offsets):
        return 0, 0
    start = offsets[line] + max(0, character)
    if end_line < 0 or end_line >= len(offsets):
        end = start
    else:
        end = offsets[end_line] + max(0, end_character)
    if end < start:
        end = start
    return start, min(end, len(text))


def parse_unused_vars(output: str) -> list[tuple[str, int, str]]:
    """(relative path, 0-based line, ident) from gc `declared and not used` lines."""
    found: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for match in _UNUSED_GC.finditer(output or ""):
        rel = match.group("rel").replace("\\", "/").lstrip("./")
        line = int(match.group("line")) - 1
        name = match.group("name")
        key = (rel, line, name)
        if line < 0 or key in seen:
            continue
        seen.add(key)
        found.append((rel, line, name))
    return found


def is_range_index_decl(line: str, name: str) -> bool:
    """True if `name` is the index ident on a `for ... range` line."""
    if not name or not name.isidentifier():
        return False
    ident = re.escape(name)
    return bool(re.search(rf"\bfor\s+{ident}\s*(,|:=|=).*?\brange\b", line))


def patch_unused_line(line: str, name: str) -> str | None:
    """Blank an unused range index only. Accumulators stay for the specialist."""
    if not is_range_index_decl(line, name):
        return None
    ident = re.escape(name)
    nl = "\n" if line.endswith("\n") else ""
    body = line[:-1] if nl else line
    rules = (
        (rf"(\bfor\s+){ident}(\s*,)", rf"\1_\2"),
        (rf"(\bfor\s+){ident}\s*:=(\s*range\b)", rf"\1_ =\2"),
    )
    for pattern, repl in rules:
        new, n = re.subn(pattern, repl, body, count=1)
        if n:
            return new + nl
    return None


def apply_unused_on_text(text: str, hits: list[tuple[int, str]]) -> str:
    if not hits:
        return text
    lines = text.splitlines(keepends=True)
    ordered = sorted(hits, key=lambda item: item[0], reverse=True)
    for line, name in ordered:
        if line < 0 or line >= len(lines):
            continue
        patched = patch_unused_line(lines[line], name)
        if patched is not None:
            lines[line] = patched
    return "".join(lines)


def _match_written(rel: str, written: list[str]) -> str | None:
    norm = rel.replace("\\", "/").lstrip("./")
    for item in written:
        got = item.replace("\\", "/")
        if got == norm or got.endswith("/" + norm) or norm.endswith("/" + got):
            return item
    return None


def apply_unused_var_fixes(root: Path, rels: list[str], output: str = "") -> list[str]:
    """Blank unused `for idx, v := range` indexes only. Named accumulators stay."""
    hits = parse_unused_vars(output)
    if not hits:
        return []
    by_rel: dict[str, list[tuple[int, str]]] = {}
    for rel, line, name in hits:
        matched = _match_written(rel, rels)
        if not matched or Path(matched).name.endswith("_test.go"):
            continue
        by_rel.setdefault(matched, []).append((line, name))
    changed: list[str] = []
    for rel, items in by_rel.items():
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        new = apply_unused_on_text(text, items)
        if new == text:
            continue
        path.write_text(new, encoding="utf-8", newline="\n")
        changed.append(rel)
    return changed


def apply_gopls_fixes(root: Path, rels: list[str], session) -> list[str]:
    """Rewrite invented selectors using gopls completion at the diagnostic site."""
    if session is None:
        return []
    changed: list[str] = []
    for rel in rels:
        path = root / rel
        if not path.is_file() or path.name.endswith("_test.go"):
            continue
        try:
            session.sync_file(rel)
        except Exception:
            continue
        try:
            diags = session.diagnostics_for(rel)
        except Exception:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        unused_hits = [
            (diag.line, name)
            for diag in diags
            if (name := unused_name_from_message(diag.message))
        ]
        new = apply_unused_on_text(text, unused_hits)
        edits: list[tuple[int, int, str]] = []
        for diag in diags:
            want = missing_method_from_message(diag.message)
            if not want:
                continue
            try:
                names = session.completions_at(rel, diag.line, diag.character)
            except Exception:
                names = []
            fix = pick_selector_fix(want, names)
            if not fix or fix == want:
                continue
            start, end = _span(new, diag.line, diag.character, diag.end_line, diag.end_character)
            snippet = new[start:end]
            if want not in snippet and snippet != want:
                # Diagnostic sometimes covers `w.WriteRune`; replace the identifier only.
                idx = snippet.rfind(want)
                if idx < 0:
                    continue
                start += idx
                end = start + len(want)
            elif snippet != want:
                idx = snippet.rfind(want)
                if idx >= 0:
                    start += idx
                    end = start + len(want)
            edits.append((start, end, fix))
        edits.sort(key=lambda item: item[0], reverse=True)
        for start, end, fix in edits:
            new = new[:start] + fix + new[end:]
        if new == text:
            continue
        path.write_text(new, encoding="utf-8", newline="\n")
        changed.append(rel)
        try:
            session.sync_file(rel)
        except Exception:
            pass
    return changed


def rank_members(want: str, names: list[str]) -> list[str]:
    if not names:
        return []
    target = (want or "").lower()

    def score(name: str) -> tuple[int, float, str]:
        lower = name.lower()
        prefix = 1 if target and (lower.startswith(target) or target.startswith(lower)) else 0
        ratio = SequenceMatcher(None, target, lower).ratio() if target else 0.0
        return (-prefix, -ratio, name)

    ordered = sorted(names, key=score)
    out: list[str] = []
    for name in ordered:
        if name not in out:
            out.append(name)
        if len(out) >= MAX_MEMBERS:
            break
    return out


def _read_sources(root: Path, rels: list[str]) -> list[str]:
    blobs: list[str] = []
    for rel in rels:
        path = root / rel
        if not path.is_file():
            continue
        try:
            blobs.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return blobs


def _merged_aliases(sources: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source in sources:
        aliases.update(import_aliases(source))
    return aliases


def _go_bin() -> str | None:
    return shutil.which("go")


def _run_go_doc(root: Path, query: str) -> str:
    go = _go_bin()
    if not go or not query.strip():
        return ""
    try:
        proc = subprocess.run(
            [go, "doc", "-short", query],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GODOC_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _package_queries(written: list[str], type_name: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for rel in written:
        parent = str(Path(rel).parent).replace("\\", "/")
        if parent in {".", ""}:
            query = type_name
        else:
            query = f"./{parent}.{type_name}"
        if query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


def _godoc_queries(miss: TypeMiss, aliases: dict[str, str], written: list[str]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    if miss.kind == "pkg":
        add(aliases.get(miss.expr) or "")
        add(miss.expr)
        return queries
    raw = _strip_type(miss.type_str)
    if not raw or any(ch in raw for ch in "[{("):
        return []
    if "." not in raw:
        for query in _package_queries(written, raw):
            add(query)
        return queries
    pkg, name = raw.rsplit(".", 1)
    path = aliases.get(pkg)
    if path:
        add(f"{path}.{name}")
    add(f"{pkg}.{name}")
    for query in _package_queries(written, name):
        add(query)
    return queries


def lookup_members(
    miss: TypeMiss,
    *,
    root: Path,
    sources: list[str],
    written: list[str] | None = None,
) -> list[str]:
    aliases = _merged_aliases(sources)
    local = members_from_source(sources, miss.type_str if miss.kind == "member" else "")
    names = list(local)
    seen = set(names)
    raw = _strip_type(miss.type_str)
    qualified = "." in raw
    imported = qualified and raw.rsplit(".", 1)[0] in aliases
    need_doc = miss.kind == "pkg" or not local or imported
    if not need_doc:
        return rank_members(miss.want, names)
    for query in _godoc_queries(miss, aliases, written or []):
        extra = members_from_godoc(
            _run_go_doc(root, query),
            funcs=miss.kind == "pkg",
        )
        for name in extra:
            if name not in seen:
                seen.add(name)
                names.append(name)
        if extra:
            break
    return rank_members(miss.want, names)


def format_member_hint(miss: TypeMiss, members: list[str]) -> str:
    ranked = rank_members(miss.want, members)
    if not ranked:
        return ""
    label = miss.type_str if miss.kind == "member" else miss.expr
    close = ranked[0]
    did = ""
    if miss.want not in ranked:
        ratio = SequenceMatcher(None, miss.want.lower(), close.lower()).ratio()
        if ratio >= 0.45 or close.lower().startswith(miss.want.lower()[:1]):
            did = f" did you mean {close}?"
    joined = ", ".join(ranked)
    if miss.kind == "pkg":
        return f"package {label} exports (pick one if it matches the spec): {joined}.{did}"
    return f"members of {label} (pick one if it matches the spec): {joined}.{did}"


def _gopls_hints(root: Path, written: list[str], session) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for rel in written:
        try:
            session.sync_file(rel)
            diags = session.diagnostics_for(rel)
        except Exception:
            continue
        for diag in diags:
            want = missing_method_from_message(diag.message)
            if not want:
                continue
            try:
                names = session.completions_at(rel, diag.line, diag.character)
            except Exception:
                names = []
            if not names:
                continue
            miss = TypeMiss(
                expr="",
                want=want,
                type_str=type_from_missing_message(diag.message) or want,
            )
            hint = format_member_hint(miss, names)
            if hint and hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return hints


def enrich_test_error(
    output: str,
    *,
    root: Path,
    written: list[str],
    session=None,
) -> str:
    """Append a short gopls-style member list for invented selectors."""
    text = (output or "").rstrip()
    misses = parse_type_errors(text)
    sources = _read_sources(root, written)
    hints: list[str] = []
    seen: set[str] = set()
    if session is not None:
        for hint in _gopls_hints(root, written, session):
            if hint not in seen:
                seen.add(hint)
                hints.append(hint)
    for miss in misses:
        members = lookup_members(miss, root=root, sources=sources, written=written)
        hint = format_member_hint(miss, members)
        if hint and hint not in seen:
            seen.add(hint)
            hints.append(hint)
    leftover = parse_unused_vars(text)
    if leftover and UNUSED_ACCUM_HINT not in text:
        text = text.rstrip() + "\n\n" + UNUSED_ACCUM_HINT
    if not hints:
        return text if leftover else (output or "")
    return (
        text
        + "\n\nLanguage server (use only these names if they match the spec):\n"
        + "\n".join(hints)
    )


def _imports_cmd() -> list[str] | None:
    goimports = shutil.which("goimports")
    if goimports:
        return [goimports, "-w"]
    gopls = shutil.which("gopls")
    if gopls:
        return [gopls, "imports", "-w"]
    return None


def apply_goimports(root: Path, rels: list[str]) -> list[str]:
    """Fill missing imports on written files. Respect later allowlist checks."""
    cmd = _imports_cmd()
    if not cmd:
        return []
    fixed: list[str] = []
    for rel in rels:
        path = root / rel
        if not path.is_file() or path.name.endswith("_test.go"):
            continue
        try:
            proc = subprocess.run(
                [*cmd, str(path)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=IMPORTS_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            fixed.append(rel)
    return fixed
