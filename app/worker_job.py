from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.farm_consult import FarmConsultError, consult
from app.worker_fs import (
    ApplyError,
    apply_files,
    format_apply_summary,
    parse_specialist_files,
    workspace_root,
)

DEFAULT_RETRIES = 3


def _go_bin() -> str | None:
    return shutil.which("go")


def packages_for_go_test(written: list[str]) -> list[str]:
    pkgs: list[str] = []
    seen: set[str] = set()
    for rel in written:
        parent = str(Path(rel).parent).replace("\\", "/")
        if parent in {".", ""}:
            pkg = "."
        else:
            pkg = "./" + parent
        if pkg not in seen:
            seen.add(pkg)
            pkgs.append(pkg)
    return pkgs or ["."]


def run_go_test(root: Path, written: list[str], *, timeout: float = 120.0) -> tuple[bool, str]:
    go = _go_bin()
    if not go:
        return True, "go test skipped: go binary not on PATH"
    pkgs = packages_for_go_test(written)
    try:
        proc = subprocess.run(
            [go, "test", *pkgs],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"go test failed to run: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    output = output.strip() or f"exit {proc.returncode}"
    ok = proc.returncode == 0
    return ok, output[-4000:]


def implement_and_apply(
    spec: str,
    *,
    constraints: str = "",
    files: str = "",
    existing_code: str = "",
    model: str | None = None,
    workspace: str = "",
    apply: bool = True,
    run_tests: bool = True,
    freeze_tests: bool = False,
    retries: int = DEFAULT_RETRIES,
    consult_fn=consult,
    go_test_fn=run_go_test,
) -> str:
    """Consult the specialist, write files on the host, optionally go test + retry."""
    attempts = max(1, int(retries))
    error = ""
    last_text = ""
    root = workspace_root(workspace) if apply else None
    for attempt in range(1, attempts + 1):
        question = spec
        if error:
            question = spec.rstrip() + "\n\nPrevious compile/test error:\n" + error
        last_text = consult_fn(
            question,
            existing_code,
            model=model,
            mode="implement",
            constraints=constraints,
            files=files,
        )
        if not apply:
            return last_text
        parsed = parse_specialist_files(last_text)
        if not parsed:
            raise ApplyError(
                "specialist returned no ### relative/path.go fenced files. "
                "Do not write the package yourself; call again with a tighter spec."
            )
        assert root is not None
        written, skipped = apply_files(root, parsed, freeze_tests=freeze_tests)
        if not written and skipped:
            return format_apply_summary(root, written, skipped=skipped) + (
                "\nNo impl files written (tests frozen). Pass freeze_tests=false to allow _test.go."
            )
        if not written:
            raise ApplyError("no files written")
        if not run_tests:
            return format_apply_summary(root, written, skipped=skipped)
        ok, output = go_test_fn(root, written)
        if ok:
            return format_apply_summary(
                root, written, skipped=skipped, test_line=f"go test: PASS\n{output}"
            )
        error = output
        if attempt >= attempts:
            return format_apply_summary(
                root,
                written,
                skipped=skipped,
                test_line=f"go test: FAIL after {attempts} attempt(s)\n{output}",
            )
    return last_text


def format_job_error(exc: BaseException) -> str:
    if isinstance(exc, (FarmConsultError, ApplyError)):
        return f"Specialist error: {exc}"
    return f"Specialist error: {exc}"
