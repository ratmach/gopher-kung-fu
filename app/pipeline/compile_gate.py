from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.models import ShareGPTExample
from app.pipeline.filters import IMPLEMENT_SKILLS
from app.worker_fs import ApplyError, parse_specialist_files
from app.worker_imports import go_imports, is_stdlib

log = logging.getLogger("gopher.compile_gate")

_GO_MISSING_LOGGED = False
GATE_MODULE = "compilegate.local"


def extract_go_files(example: ShareGPTExample) -> list[tuple[str, str]]:
    if len(example.conversations) < 2:
        return []
    gpt = example.conversations[-1].value
    try:
        parsed = parse_specialist_files(gpt)
    except ApplyError:
        return []
    return [(item.rel, item.body) for item in parsed]


def _stdlib_only(body: str) -> bool:
    return all(is_stdlib(path) for path in go_imports(body))


def compile_gate_example(example: ShareGPTExample, *, go_bin: str | None | bool = None) -> tuple[bool, str]:
    """Keep write/debug Go rows that `go test` can compile. Skip the gate if go is missing."""
    skill = str(example.meta.get("skill", ""))
    if skill and skill not in IMPLEMENT_SKILLS:
        return True, "skip: not implement"
    files = extract_go_files(example)
    if not files:
        return False, "no ### path.go fences"
    if any(not _stdlib_only(body) for _, body in files):
        return False, "non-stdlib import"
    if go_bin is False:
        return True, "skip: go missing"
    binary = go_bin if isinstance(go_bin, str) else shutil.which("go")
    if not binary:
        global _GO_MISSING_LOGGED
        if not _GO_MISSING_LOGGED:
            log.warning("compile-gate skipped: go binary not on PATH")
            _GO_MISSING_LOGGED = True
        return True, "skip: go missing"
    with tempfile.TemporaryDirectory(prefix="gopher-gate-") as raw:
        root = Path(raw)
        (root / "go.mod").write_text(f"module {GATE_MODULE}\n\ngo 1.22\n", encoding="utf-8")
        for rel, body in files:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8", newline="\n")
        try:
            proc = subprocess.run(
                [binary, "test", "./..."],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"go test failed to run: {exc}"
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            return False, output[-500:] or f"exit {proc.returncode}"
    return True, "ok"
