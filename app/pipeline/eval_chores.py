from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from app.paths import ROOT
from app.worker_job import implement_and_apply

CHORES_DIR = ROOT / "tests" / "eval_chores"


def load_chore(folder: Path) -> dict[str, Any]:
    data = json.loads((folder / "chore.json").read_text(encoding="utf-8"))
    data["dir"] = folder
    return data


def list_chores(root: Path | None = None) -> list[dict[str, Any]]:
    base = root or CHORES_DIR
    if not base.is_dir():
        return []
    chores: list[dict[str, Any]] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        if (folder / "chore.json").is_file():
            chores.append(load_chore(folder))
    return chores


def _stage_workspace(chore: dict[str, Any], dest: Path) -> None:
    (dest / "go.mod").write_text("module evalchore.local\n\ngo 1.22\n", encoding="utf-8")
    hidden = chore["dir"] / "hidden"
    impl = Path(str(chore.get("files") or "impl.go"))
    test_dest = dest / impl.parent / (impl.stem + "_test.go")
    test_dest.parent.mkdir(parents=True, exist_ok=True)
    if hidden.is_dir():
        tests = list(hidden.glob("*_test.go"))
        if tests:
            test_dest.write_text(tests[0].read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def run_chore(
    chore: dict[str, Any],
    *,
    consult_fn,
    workspace: Path,
    retries: int = 3,
) -> dict[str, Any]:
    _stage_workspace(chore, workspace)
    attempts = {"n": 0}

    def counted(*args, **kwargs):
        attempts["n"] += 1
        return consult_fn(*args, **kwargs)

    summary = implement_and_apply(
        str(chore.get("spec") or ""),
        constraints=str(chore.get("constraints") or "stdlib only, no CGO"),
        files=str(chore.get("files") or ""),
        existing_code=str(chore.get("existing_code") or ""),
        workspace=str(workspace),
        freeze_tests=True,
        retries=retries,
        allowed_imports=list(chore.get("allowed_imports") or []),
        forbidden_imports=list(chore.get("forbidden_imports") or ["C"]),
        consult_fn=counted,
    )
    passed = "go test: PASS" in summary
    return {
        "id": chore.get("id") or workspace.name,
        "passed": passed,
        "pass_at_1": passed and attempts["n"] == 1,
        "pass_at_3": passed and attempts["n"] <= retries,
        "attempts": attempts["n"],
        "summary": summary,
    }


def run_suite(
    chores: list[dict[str, Any]] | None = None,
    *,
    consult_fn: Callable[..., str],
    retries: int = 3,
) -> dict[str, Any]:
    rows = chores if chores is not None else list_chores()
    results: list[dict[str, Any]] = []
    for chore in rows:
        with TemporaryDirectory(prefix="gopher-chore-") as raw:
            results.append(run_chore(chore, consult_fn=consult_fn, workspace=Path(raw), retries=retries))
    n = max(1, len(results))
    return {
        "chores": len(results),
        "pass_at_1": sum(1 for row in results if row["pass_at_1"]) / n,
        "pass_at_3": sum(1 for row in results if row["pass_at_3"]) / n,
        "results": results,
    }


def main() -> None:
    from app.farm_consult import consult

    parser = argparse.ArgumentParser(description="Held-out implement chores (hidden tests)")
    parser.add_argument("--chores", type=Path, default=CHORES_DIR)
    args = parser.parse_args()
    report = run_suite(list_chores(args.chores), consult_fn=consult)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for row in report["results"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{row['id']}: {status} attempts={row['attempts']}")
    if report["pass_at_3"] < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
