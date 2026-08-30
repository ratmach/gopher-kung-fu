from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from app.farm_consult import FarmConsultError, consult, default_model
from app.worker_fs import (
    ApplyError,
    apply_files,
    format_apply_summary,
    map_parsed_to_wanted,
    parse_specialist_files,
    parse_wanted_files,
    previous_attempt_blob,
    record_write_hashes,
    workspace_root,
)
from app.worker_catalog import build_neighbor_catalog
from app.worker_gopls import start_gopls
from app.worker_imports import check_written_imports, resolve_allowlists
from app.worker_intel import apply_goimports, apply_gopls_fixes, apply_unused_var_fixes, enrich_test_error

DEFAULT_RETRIES = 3
_GO_FILE_IN_ERROR = re.compile(r"((?:[A-Za-z0-9_./\\-]|\\\\)+\.go):\d+")
_COMPILE_LOC = re.compile(r"\.go:\d+:\d+:")


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


def go_files_in_test_output(output: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in _GO_FILE_IN_ERROR.findall(output or ""):
        rel = raw.replace("\\", "/").lstrip("./")
        if rel not in seen:
            seen.add(rel)
            found.append(rel)
    return found


FROZEN_TEST_COMPILE_HINT = (
    "Compiler cited frozen *_test.go. Do not edit tests. "
    "Patch this impl so those references compile (missing types, funcs, or arity)."
)


def error_only_in_frozen_tests(output: str, *, freeze_tests: bool) -> bool:
    """True when go test compile errors only name *_test.go files.

    That is usually a missing impl API (`undefined: Parse` in the test), not a
    broken suite. Keep repairing the impl. Do not treat this as a stop.
    """
    if not freeze_tests:
        return False
    text = output or ""
    if "--- FAIL:" in text:
        return False
    names = go_files_in_test_output(text)
    if not names:
        return False
    if not all(Path(name).name.endswith("_test.go") for name in names):
        return False
    return "[build failed]" in text or bool(_COMPILE_LOC.search(text))


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


def wanted_impl_paths(files: str, *, freeze_tests: bool) -> list[str]:
    wanted = parse_wanted_files(files)
    tests = [rel for rel in wanted if Path(rel).name.endswith("_test.go")]
    if freeze_tests and tests:
        raise ApplyError("frozen job cannot request test files: " + ", ".join(tests))
    return wanted


def run_job_files(files: str) -> list[str]:
    wanted = wanted_impl_paths(files, freeze_tests=True)
    impl = [rel for rel in wanted if not Path(rel).name.endswith("_test.go")]
    if len(impl) != 1:
        raise ApplyError("run_job expects exactly one impl .go path in files=")
    return wanted


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
    freeze_tests: bool = True,
    retries: int = DEFAULT_RETRIES,
    allowed_imports: list[str] | None = None,
    forbidden_imports: list[str] | None = None,
    consult_fn=consult,
    go_test_fn=run_go_test,
    gopls_factory=None,
) -> str:
    """Consult the specialist, write files on the host, optionally go test + repair-retry."""
    attempts = max(1, int(retries))
    root = workspace_root(workspace) if apply else None
    wanted = wanted_impl_paths(files, freeze_tests=freeze_tests)
    catalog = ""
    if root is not None and wanted:
        catalog = build_neighbor_catalog(
            root,
            wanted,
            spec=spec,
            existing_code=existing_code,
            files=files,
        )
    allowed, forbidden = resolve_allowlists(
        model,
        allowed_imports=allowed_imports,
        forbidden_imports=forbidden_imports,
        constraints=constraints,
    )
    factory = start_gopls if gopls_factory is None else gopls_factory
    session = factory(root) if root is not None else None
    try:
        return _run_implement_attempts(
            spec,
            constraints=constraints,
            files=files,
            existing_code=existing_code,
            model=model,
            apply=apply,
            run_tests=run_tests,
            freeze_tests=freeze_tests,
            attempts=attempts,
            allowed=allowed,
            forbidden=forbidden,
            catalog=catalog,
            root=root,
            wanted=wanted,
            consult_fn=consult_fn,
            go_test_fn=go_test_fn,
            session=session,
        )
    finally:
        closer = getattr(session, "close", None)
        if callable(closer):
            closer()


def _run_implement_attempts(
    spec: str,
    *,
    constraints: str,
    files: str,
    existing_code: str,
    model: str | None,
    apply: bool,
    run_tests: bool,
    freeze_tests: bool,
    attempts: int,
    allowed: list[str],
    forbidden: list[str],
    catalog: str,
    root,
    wanted: list[str],
    consult_fn,
    go_test_fn,
    session,
) -> str:
    error = ""
    previous_files = ""
    last_text = ""
    last_written: list[str] = []
    last_skipped: list[str] = []
    last_extra: list[str] = []
    last_missing: list[str] = []
    for attempt in range(1, attempts + 1):
        last_text = consult_fn(
            spec,
            existing_code,
            model=model,
            mode="implement",
            constraints=constraints,
            files=files,
            previous_files=previous_files,
            test_error=error,
            neighbor_apis=catalog,
        )
        if not apply:
            return last_text
        parsed = parse_specialist_files(last_text)
        if not parsed:
            raise ApplyError(
                "specialist returned no ### relative/path.go fenced files. "
                "Do not write the package yourself; call again with a tighter spec."
            )
        mapped, extras = map_parsed_to_wanted(parsed, wanted)
        if wanted and not mapped:
            got = ", ".join(item.rel for item in parsed) or "(none)"
            raise ApplyError(
                "specialist wrote no requested files. "
                f"wanted: {', '.join(wanted)}; got: {got}"
            )
        to_write = mapped if wanted else parsed
        assert root is not None
        written, skipped = apply_files(root, to_write, freeze_tests=freeze_tests)
        apply_goimports(root, written)
        if session is not None:
            fixed = apply_gopls_fixes(root, written, session)
            if fixed:
                apply_goimports(root, fixed)
        record_write_hashes(root, written, attempt=attempt)
        mapped_rels = {item.rel for item in mapped} if wanted else {item.rel for item in parsed}
        missing = [rel for rel in wanted if rel not in mapped_rels]
        last_written, last_skipped, last_extra, last_missing = written, skipped, extras, missing
        if not written and skipped:
            return format_apply_summary(root, written, skipped=skipped, skipped_extra=extras, missing=missing) + (
                "\nNo impl files written (tests frozen). Pass freeze_tests=false to allow _test.go."
            )
        if not written:
            raise ApplyError("no files written")
        hits = check_written_imports(
            root,
            written,
            allowed_imports=allowed,
            forbidden_imports=forbidden,
        )
        if hits:
            return format_apply_summary(
                root,
                written,
                skipped=skipped,
                skipped_extra=extras,
                missing=missing,
                test_line=(
                    "FAIL: import not on cartridge allowlist: "
                    + ", ".join(hits)
                    + ". Do not retry; change the job or implement yourself."
                ),
            )
        if not run_tests:
            return format_apply_summary(
                root, written, skipped=skipped, skipped_extra=extras, missing=missing
            )
        ok, output = go_test_fn(root, written)
        if not ok:
            unused = apply_unused_var_fixes(root, written, output)
            if unused:
                apply_goimports(root, unused)
                record_write_hashes(root, unused, attempt=attempt)
                ok, output = go_test_fn(root, written)
        if not ok:
            output = enrich_test_error(output, root=root, written=written, session=session)
        if ok:
            return format_apply_summary(
                root,
                written,
                skipped=skipped,
                skipped_extra=extras,
                missing=missing,
                test_line=f"go test: PASS\n{output}",
            )
        if error_only_in_frozen_tests(output, freeze_tests=freeze_tests):
            if FROZEN_TEST_COMPILE_HINT not in output:
                output = output.rstrip() + "\n\n" + FROZEN_TEST_COMPILE_HINT
        error = output
        previous_files = previous_attempt_blob(root, written)
        if attempt >= attempts:
            note = "repair used previous file + compiler error" if attempts > 1 else "single attempt"
            return format_apply_summary(
                root,
                written,
                skipped=skipped,
                skipped_extra=extras,
                missing=missing,
                test_line=f"go test: FAIL after {attempts} attempt(s) ({note})\n{output}",
            )
    return format_apply_summary(
        root or Path("."),
        last_written,
        skipped=last_skipped,
        skipped_extra=last_extra,
        missing=last_missing,
        test_line=last_text,
    )


def run_job(
    spec: str,
    *,
    constraints: str = "",
    files: str = "",
    existing_code: str = "",
    workspace: str = "",
    model: str | None = None,
    consult_fn=consult,
    go_test_fn=run_go_test,
) -> str:
    """Supervisor entry: freeze tests, one impl file, cartridge allowlist, then repair loop."""
    run_job_files(files)
    mid = (model or default_model()).strip() or default_model()
    return implement_and_apply(
        spec,
        constraints=constraints,
        files=files,
        existing_code=existing_code,
        model=mid,
        workspace=workspace,
        apply=True,
        run_tests=True,
        freeze_tests=True,
        consult_fn=consult_fn,
        go_test_fn=go_test_fn,
    )


def format_job_error(exc: BaseException) -> str:
    if isinstance(exc, (FarmConsultError, ApplyError)):
        return f"Specialist error: {exc}"
    return f"Specialist error: {exc}"
