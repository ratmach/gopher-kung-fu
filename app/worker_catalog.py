from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from app.worker_imports import go_imports, is_stdlib

MAX_PACKAGES = 8
MAX_CHARS = 4000
RETRIEVE_K = 6
MIN_CHUNKS = 3
NGRAM = 3
INDEX_REL = ".gopher/catalog_index.json"
_SKIP_DIR = {"vendor", ".git", "testdata", ".gopher", "node_modules"}

_MODULE = re.compile(r"(?m)^module\s+(\S+)")
_FUNC = re.compile(r"(?m)^func\s+(?:\(([^)]+)\)\s+)?([A-Za-z_]\w*)\s*\(")
_TYPE_STRUCT = re.compile(r"(?m)^type\s+([A-Za-z_]\w*)\s+struct\s*\{")
_TYPE_IFACE = re.compile(r"(?m)^type\s+([A-Za-z_]\w*)\s+interface\s*\{")
_TYPE_ALIAS = re.compile(
    r"(?m)^type\s+([A-Za-z_]\w*)\s+(?!struct\b|interface\b)([^\n]+)"
)


@dataclass(frozen=True)
class Export:
    pkg: str
    file: str
    kind: str
    name: str
    text: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.pkg, self.name)


def module_path(root: Path) -> str:
    path = root / "go.mod"
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _MODULE.search(text)
    return (match.group(1) if match else "").strip()


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def package_dir(impl_rel: str) -> str:
    parent = str(Path(impl_rel).parent).replace("\\", "/")
    if parent in {".", ""}:
        return "."
    return parent


def import_to_rel(mod: str, import_path: str) -> str | None:
    if not mod or not import_path:
        return None
    if import_path == mod:
        return "."
    prefix = mod + "/"
    if import_path.startswith(prefix):
        rel = import_path[len(prefix) :].strip("/")
        return rel or "."
    return None


def _read_go(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _go_files_in(root: Path, rel_dir: str) -> list[Path]:
    folder = root if rel_dir in {".", ""} else root / rel_dir
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.go") if p.is_file())


def neighbor_import_paths(root: Path, wanted_rels: list[str]) -> list[str]:
    """Same-module import dirs referenced by the impl package (tests + siblings)."""
    mod = module_path(root)
    if not mod or not wanted_rels:
        return []
    self_dirs = {package_dir(rel) for rel in wanted_rels}
    found: list[str] = []
    seen: set[str] = set()
    for rel in wanted_rels:
        for path in _go_files_in(root, package_dir(rel)):
            for item in go_imports(_read_go(path)):
                if item == "C" or is_stdlib(item):
                    continue
                mapped = import_to_rel(mod, item)
                if not mapped or mapped in self_dirs:
                    continue
                dest = root if mapped == "." else root / mapped
                if not dest.is_dir():
                    continue
                if mapped not in seen:
                    seen.add(mapped)
                    found.append(mapped)
    return found


def _matching_pair(src: str, open_idx: int, opener: str, closer: str) -> int:
    depth = 0
    for i in range(open_idx, len(src)):
        ch = src[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _exported(name: str) -> bool:
    return bool(name) and name[:1].isupper()


def extract_exports(source: str, *, pkg: str, file: str) -> list[Export]:
    text = source or ""
    out: list[Export] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: str, blob: str) -> None:
        blob = blob.strip()
        if not blob or (kind, name) in seen or not _exported(name):
            return
        seen.add((kind, name))
        out.append(Export(pkg=pkg, file=file, kind=kind, name=name, text=blob))

    for match in _TYPE_STRUCT.finditer(text):
        name = match.group(1)
        close = _matching_pair(text, match.end() - 1, "{", "}")
        if close < 0:
            continue
        add("struct", name, text[match.start() : close + 1])
    for match in _TYPE_IFACE.finditer(text):
        name = match.group(1)
        close = _matching_pair(text, match.end() - 1, "{", "}")
        if close < 0:
            continue
        add("interface", name, text[match.start() : close + 1])
    for match in _TYPE_ALIAS.finditer(text):
        name = match.group(1)
        add("type", name, text[match.start() : match.end()].rstrip())
    for match in _FUNC.finditer(text):
        name = match.group(2)
        if name == "init":
            continue
        paren_open = match.end() - 1
        paren_close = _matching_pair(text, paren_open, "(", ")")
        if paren_close < 0:
            continue
        rest = text[paren_close + 1 :]
        brace = rest.find("{")
        newline = rest.find("\n")
        if brace >= 0 and (newline < 0 or brace < newline):
            sig = text[match.start() : paren_close + 1 + brace].rstrip()
        else:
            end = newline if newline >= 0 else len(rest)
            sig = text[match.start() : paren_close + 1 + end].rstrip()
        kind = "method" if match.group(1) else "func"
        add(kind, name, sig)
    return out


def _iter_module_go(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*.go"):
        if not path.is_file() or path.name.endswith("_test.go"):
            continue
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _SKIP_DIR for part in rel_parts):
            continue
        found.append(path)
    return sorted(found)


def exports_from_dir(root: Path, rel_dir: str) -> list[Export]:
    pkg = rel_dir if rel_dir not in {"", "."} else "."
    out: list[Export] = []
    for path in _go_files_in(root, rel_dir):
        if path.name.endswith("_test.go"):
            continue
        out.extend(extract_exports(_read_go(path), pkg=pkg, file=_rel_posix(path, root)))
    return out


def all_module_exports(root: Path) -> list[Export]:
    out: list[Export] = []
    for path in _iter_module_go(root):
        rel = _rel_posix(path, root)
        pkg = str(Path(rel).parent).replace("\\", "/")
        if pkg in {"", "."}:
            pkg = "."
        out.extend(extract_exports(_read_go(path), pkg=pkg, file=rel))
    return out


def _char_ngrams(text: str, n: int = NGRAM) -> dict[str, int]:
    folded = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not folded:
        return {}
    if len(folded) < n:
        return {folded: 1}
    counts: dict[str, int] = {}
    for i in range(len(folded) - n + 1):
        gram = folded[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _tfidf_vec(counts: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {
        gram: (freq / total) * idf[gram]
        for gram, freq in counts.items()
        if gram in idf
    }


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values())) or 1.0


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(val * b.get(key, 0.0) for key, val in a.items())
    return dot / (_norm(a) * _norm(b))


def _chunk_doc(item: Export) -> str:
    return f"{item.name} {item.pkg} {item.text}"


def _fit_index(chunks: list[Export]) -> dict:
    docs = [_char_ngrams(_chunk_doc(item)) for item in chunks]
    n = len(docs)
    df: dict[str, int] = {}
    for counts in docs:
        for gram in counts:
            df[gram] = df.get(gram, 0) + 1
    idf = {gram: math.log((n + 1) / (freq + 1)) + 1.0 for gram, freq in df.items()}
    rows = []
    for item, counts in zip(chunks, docs):
        vec = _tfidf_vec(counts, idf)
        rows.append(
            {
                "pkg": item.pkg,
                "file": item.file,
                "kind": item.kind,
                "name": item.name,
                "text": item.text,
                "vector": {k: round(v, 6) for k, v in vec.items()},
            }
        )
    return {"idf": {k: round(v, 6) for k, v in idf.items()}, "chunks": rows}


def _source_mtimes(root: Path) -> dict[str, float]:
    return {_rel_posix(path, root): path.stat().st_mtime for path in _iter_module_go(root)}


def _index_path(root: Path) -> Path:
    return root / INDEX_REL


def _index_stale(root: Path, payload: dict) -> bool:
    stored = payload.get("mtimes") or {}
    current = _source_mtimes(root)
    if set(stored) != set(current):
        return True
    for rel, mtime in current.items():
        try:
            old = float(stored[rel])
        except (KeyError, TypeError, ValueError):
            return True
        if abs(old - mtime) > 1e-6:
            return True
    return False


def load_or_build_index(root: Path) -> dict:
    path = _index_path(root)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        else:
            if isinstance(payload, dict) and not _index_stale(root, payload):
                return payload
    chunks = all_module_exports(root)
    payload = _fit_index(chunks)
    payload["mtimes"] = _source_mtimes(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def retrieve_extras(
    index: dict,
    query: str,
    *,
    exclude: set[tuple[str, str]],
    k: int = RETRIEVE_K,
) -> list[Export]:
    chunks = index.get("chunks") or []
    if len(chunks) < MIN_CHUNKS:
        return []
    idf = index.get("idf") or {}
    qvec = _tfidf_vec(_char_ngrams(query), idf)
    if not qvec:
        return []
    scored: list[tuple[float, Export]] = []
    for row in chunks:
        pkg = str(row.get("pkg") or "")
        name = str(row.get("name") or "")
        if (pkg, name) in exclude:
            continue
        text = str(row.get("text") or "")
        if not text:
            continue
        vec = row.get("vector") or {}
        score = _cosine(qvec, {str(g): float(v) for g, v in vec.items()})
        if score <= 0:
            continue
        scored.append(
            (
                score,
                Export(
                    pkg=pkg,
                    file=str(row.get("file") or ""),
                    kind=str(row.get("kind") or "func"),
                    name=name,
                    text=text,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].pkg, item[1].name))
    out: list[Export] = []
    seen: set[tuple[str, str]] = set()
    for _, item in scored:
        if item.key in seen:
            continue
        seen.add(item.key)
        out.append(item)
        if len(out) >= k:
            break
    return out


def _pkg_block(pkg: str, exports: list[Export]) -> str:
    body = "\n".join(item.text for item in exports)
    return f"# {pkg}\n{body}".rstrip()


def format_catalog(
    exact: list[Export],
    extras: list[Export],
    *,
    budget: int = MAX_CHARS,
    max_packages: int = MAX_PACKAGES,
) -> str:
    grouped: dict[str, list[Export]] = {}
    for item in exact:
        grouped.setdefault(item.pkg, []).append(item)
    parts: list[str] = []
    used = 0
    for i, (pkg, rows) in enumerate(grouped.items()):
        if i >= max_packages:
            break
        block = _pkg_block(pkg, rows)
        extra_nl = 2 if parts else 0
        if used and used + extra_nl + len(block) > budget:
            break
        if parts:
            parts.append("")
        parts.append(block)
        used = len("\n".join(parts))
    remain = budget - used
    extra_parts: list[str] = []
    if extras and remain > 24:
        header = "# also relevant"
        extra_parts.append(header)
        filled = len(header)
        last_pkg = ""
        for item in extras:
            prefix = f"# {item.pkg}\n" if item.pkg != last_pkg else ""
            chunk = prefix + item.text
            add = len(chunk) + 1
            if filled + add > remain:
                break
            extra_parts.append(chunk)
            filled += add
            last_pkg = item.pkg
        if len(extra_parts) > 1:
            if parts:
                parts.append("")
            parts.append("\n".join(extra_parts))
    return "\n".join(parts).strip()


def build_neighbor_catalog(
    root: Path,
    wanted_rels: list[str],
    *,
    spec: str = "",
    existing_code: str = "",
    files: str = "",
) -> str:
    neighbors = neighbor_import_paths(root, wanted_rels)
    exact: list[Export] = []
    for rel_dir in neighbors:
        exact.extend(exports_from_dir(root, rel_dir))
    exclude = {item.key for item in exact}
    query = "\n".join(part for part in (spec, files, existing_code) if (part or "").strip())
    extras: list[Export] = []
    if query.strip():
        index = load_or_build_index(root)
        extras = retrieve_extras(index, query, exclude=exclude)
    return format_catalog(exact, extras)
