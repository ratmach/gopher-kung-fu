from __future__ import annotations

import logging
import sys

import httpx

from app.farm_consult import FarmConsultError, consult, default_model, farm_url
from app.worker_job import format_job_error, implement_and_apply, run_job as run_supervised_job

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'MCP SDK missing. In this venv run: pip install -e ".[mcp]"'
    ) from exc


def _windows_stdio_lf() -> None:
    """MCP JSON-RPC is newline-delimited. Windows text wrappers otherwise emit CRLF."""
    if sys.platform != "win32":
        return
    import mcp.server.stdio as mcp_stdio

    class LFWrapper(mcp_stdio._UnownedTextWrapper):
        def __init__(self, *args, **kwargs):
            kwargs["newline"] = ""
            super().__init__(*args, **kwargs)

    mcp_stdio._UnownedTextWrapper = LFWrapper  # type: ignore[misc]


_windows_stdio_lf()

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="gopher-mcp: %(message)s")
log = logging.getLogger("gopher-mcp")

mcp = MCPServer(
    "gopher-kungfu",
    instructions=(
        "Local Go specialist. Call run_job BEFORE writing production Go files. "
        "Pass workspace (Go module root), spec, constraints, and one impl path. "
        "The MCP host writes files and runs go test; you only get a summary. "
        "Do not copy specialist source into your own write tool. "
        "Use implement_go only as a lower-level escape. "
        "Use ask_gopher_kungfu only for review/debug of existing snippets. "
        "Never forward the IDE system prompt or other tool schemas."
    ),
    log_level="WARNING",
)


@mcp.tool()
def run_job(
    spec: str,
    constraints: str = "",
    files: str = "",
    existing_code: str = "",
    workspace: str = "",
    model: str = "",
) -> str:
    """Supervisor: one impl file, tests frozen, cartridge import allowlist, then repair loop.

    Preferred orchestrator entry. Pass:
    - spec: behavior and tests that must pass
    - constraints: allowed modules, no-CGO, layout, names
    - files: exactly one relative impl .go path (not *_test.go)
    - existing_code: neighboring signatures only
    - workspace: absolute Go module root (or set GOPHER_WORKSPACE)

    Returns a short summary (paths + go test), not the file bodies.
    Tests already in the workspace stay frozen. Off-allowlist imports fail
    the job without further specialist retries.
    """
    try:
        return run_supervised_job(
            spec,
            constraints=constraints,
            files=files,
            existing_code=existing_code,
            workspace=workspace,
            model=model.strip() or None,
        )
    except Exception as exc:
        log.error("%s", exc)
        return format_job_error(exc)


@mcp.tool()
def implement_go(
    spec: str,
    constraints: str = "",
    files: str = "",
    existing_code: str = "",
    workspace: str = "",
    apply: bool = True,
    run_tests: bool = True,
    freeze_tests: bool = True,
    model: str = "",
) -> str:
    """Offload Go implementation. The MCP host writes files — do not copy source.

    Call BEFORE you write production Go yourself. Pass:
    - spec: behavior and tests that must pass
    - constraints: allowed modules, no-CGO, layout, names
    - files: newline-separated relative .go paths
    - existing_code: neighboring signatures only
    - workspace: absolute Go module root (or set GOPHER_WORKSPACE)

    Returns a short summary (paths + go test), not the file bodies.
    Set apply=false only to inspect raw specialist text.
    freeze_tests defaults true (skips overwriting *_test.go). Pass false only if the specialist must write tests.
    On test failure the host retries up to 3 times, sending the failing
    file body plus the compiler/test error (a repair, not a fresh sample).
    Only paths in files= are written (basename remap allowed).
    Prefer one file or one package per call.
    """
    try:
        return implement_and_apply(
            spec,
            constraints=constraints,
            files=files,
            existing_code=existing_code,
            model=model.strip() or None,
            workspace=workspace,
            apply=apply,
            run_tests=run_tests,
            freeze_tests=freeze_tests,
        )
    except Exception as exc:
        log.error("%s", exc)
        return format_job_error(exc)


@mcp.tool()
def ask_gopher_kungfu(question: str, code: str = "", model: str = "") -> str:
    """Review or debug existing Go/gin/SQL/concurrency code with the specialist.

    Not for green-field implementation — use run_job for that.
    Pass a focused question and the relevant snippet only.
    """
    try:
        return consult(question, code, model=model.strip() or None, mode="ask")
    except FarmConsultError as exc:
        log.error("%s", exc)
        return f"Specialist error: {exc}"


@mcp.tool()
def list_specialists() -> str:
    """List specialist model ids currently on the local farm."""
    models_url = farm_url().removesuffix("/chat/completions") + "/models"
    try:
        response = httpx.get(models_url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        return f"Specialist error: cannot list farm models at {models_url}: {exc}"
    ids = []
    for item in data.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    if not ids:
        return "No specialists on the farm. Export a cartridge first."
    default = default_model()
    lines = [f"{mid} (default)" if mid == default else mid for mid in ids]
    return "\n".join(lines)


def main() -> None:
    print("gopher-mcp listening on stdio", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
