from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_FARM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "gopher-kungfu"
DEFAULT_MAX_TOKENS = 2048
IMPLEMENT_MAX_TOKENS = 6144

IMPLEMENT_CONTRACT = """You are the Go specialist IMPLEMENTER, not an advisor.
Return production code. Do not give architecture essays, package shopping lists, or sketches.

For each file, use exactly this shape:

### relative/path.go
```go
package ...
```

Rules:
- Complete files only. No TODOs, no "the rest is similar", no omitted functions.
- Honor constraints exactly (allowed modules, CGO, layout, names).
- Do not add dependencies the spec did not allow.
- If tests are requested, include complete `_test.go` files.
- After the files, at most two sentences. No preamble before the first file.
"""


class FarmConsultError(RuntimeError):
    pass


def farm_url() -> str:
    return os.environ.get("GOPHER_FARM_URL", DEFAULT_FARM_URL).rstrip("/")


def default_model() -> str:
    return os.environ.get("GOPHER_FARM_MODEL", DEFAULT_MODEL)


def compose_user_content(
    question: str,
    code: str = "",
    *,
    mode: str = "ask",
    constraints: str = "",
    files: str = "",
) -> str:
    q = (question or "").strip()
    if not q:
        raise FarmConsultError("question is required")
    parts: list[str] = []
    if (mode or "ask").lower() == "implement":
        parts.append(IMPLEMENT_CONTRACT)
        parts.append("Spec:\n" + q)
        extra = (constraints or "").strip()
        if extra:
            parts.append("Constraints:\n" + extra)
        wanted = (files or "").strip()
        if wanted:
            parts.append("Files to write:\n" + wanted)
    else:
        parts.append(q)
    snippet = (code or "").strip()
    if snippet:
        parts.append("Existing code:\n```go\n" + snippet + "\n```")
    return "\n\n".join(parts)


def chat_payload(
    question: str,
    code: str = "",
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    mode: str = "ask",
    constraints: str = "",
    files: str = "",
) -> dict[str, Any]:
    kind = (mode or "ask").lower()
    if max_tokens is None:
        max_tokens = IMPLEMENT_MAX_TOKENS if kind == "implement" else DEFAULT_MAX_TOKENS
    return {
        "model": (model or default_model()).strip() or default_model(),
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": compose_user_content(
                    question, code, mode=kind, constraints=constraints, files=files
                ),
            }
        ],
    }


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return str(content).strip()


def parse_chat_completion(data: dict[str, Any]) -> str:
    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        raise FarmConsultError(str(err["message"]))
    if isinstance(err, str) and err.strip():
        raise FarmConsultError(err.strip())
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise FarmConsultError(f"farm returned no choices: {data}")
    message = choices[0].get("message") or {}
    text = _message_text(message.get("content"))
    if not text:
        raise FarmConsultError("farm returned empty content")
    return text


def consult(
    question: str,
    code: str = "",
    *,
    model: str | None = None,
    url: str | None = None,
    max_tokens: int | None = None,
    mode: str = "ask",
    constraints: str = "",
    files: str = "",
    timeout: float = 180.0,
    client: httpx.Client | None = None,
) -> str:
    payload = chat_payload(
        question,
        code,
        model=model,
        max_tokens=max_tokens,
        mode=mode,
        constraints=constraints,
        files=files,
    )
    target = (url or farm_url()).rstrip("/")
    own = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.post(target, json=payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise FarmConsultError(
                f"farm HTTP {response.status_code}: {response.text[:500]}"
            ) from exc
        if response.status_code >= 400:
            if isinstance(data, dict):
                try:
                    parse_chat_completion(data)
                except FarmConsultError:
                    raise
            raise FarmConsultError(f"farm HTTP {response.status_code}: {data}")
        return parse_chat_completion(data if isinstance(data, dict) else {})
    except httpx.RequestError as exc:
        raise FarmConsultError(
            f"cannot reach farm at {target} — is cartridge-server running? {exc}"
        ) from exc
    finally:
        if own:
            http.close()
