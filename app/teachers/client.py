from __future__ import annotations

import ast
import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.models import Project
from app.teachers.presets import preset_by_id

JSON_FENCE = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
StatusFn = Callable[[dict[str, Any]], Awaitable[None]]
CheckFn = Callable[[], None]


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchChatRequest:
    custom_id: str
    system: str
    user: str
    temperature: float = 0.7


@dataclass(frozen=True)
class BatchChatResult:
    custom_id: str
    text: str | None
    error: str | None = None


def teacher_supports_batch(project: Project) -> bool:
    if project.teacher_preset == "openrouter":
        return True
    url = (project.teacher_base_url or "").lower()
    return "openrouter.ai" in url


def strip_batch_suffix(model: str) -> str:
    return model[: -len(":batch")] if model.endswith(":batch") else model


def batch_api_url(chat_base_url: str) -> str:
    parsed = urlparse(chat_base_url)
    if not parsed.scheme or not parsed.netloc:
        raise TeacherError("cannot derive OpenRouter batch URL from teacher base URL")
    return f"{parsed.scheme}://{parsed.netloc}/api/beta/batches"


class TeacherClient:
    def __init__(self, project: Project, api_key: str, timeout: float = 120.0) -> None:
        self.project = project
        self.api_key = api_key
        self.timeout = timeout
        preset = preset_by_id(project.teacher_preset)
        self.base_url = (project.teacher_base_url or preset["base_url"]).rstrip("/")
        self.model = project.teacher_model or preset["model"]
        if not self.model:
            raise TeacherError("teacher model name is required")
        if not self.base_url:
            raise TeacherError("teacher base URL is required")

    @property
    def supports_batch(self) -> bool:
        return teacher_supports_batch(self.project)

    @property
    def batch_model(self) -> str:
        return strip_batch_suffix(self.model)

    @property
    def batch_url(self) -> str:
        return batch_api_url(self.base_url)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.4,
        http: httpx.AsyncClient | None = None,
    ) -> Any:
        text = await self.chat(system, user, temperature=temperature, json_mode=True, http=http)
        return coerce_json(text)

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
        http: httpx.AsyncClient | None = None,
    ) -> Any:
        url = f"{self.base_url}/chat/completions"
        payload = _chat_body(self.model, system, user, temperature, json_mode)
        if http is None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await self._post_chat(
                    client, url, payload, system, user, temperature, json_mode
                )
        return await self._post_chat(http, url, payload, system, user, temperature, json_mode)

    async def _post_chat(
        self,
        http: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        system: str,
        user: str,
        temperature: float,
        json_mode: bool,
    ) -> Any:
        try:
            response = await http.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise TeacherError(f"teacher request failed: {exc}") from exc
        if response.status_code >= 400:
            if json_mode and response.status_code not in {401, 403, 404}:
                return await self.chat(
                    system, user, temperature=temperature, json_mode=False, http=http
                )
            raise TeacherError(f"teacher HTTP {response.status_code}: {response.text[:800]}")
        return completion_content(response.json())

    async def run_chat_batch(
        self,
        requests: list[BatchChatRequest],
        *,
        json_mode: bool = True,
        poll_seconds: float = 8.0,
        check: CheckFn | None = None,
        on_status: StatusFn | None = None,
    ) -> list[BatchChatResult]:
        if not self.supports_batch:
            raise TeacherError(
                "OpenRouter batch needs an OpenRouter teacher (preset or openrouter.ai base URL)"
            )
        if not requests:
            return []
        payload = _batch_payload(self.batch_model, requests, json_mode=json_mode)
        timeout = httpx.Timeout(30.0, read=max(self.timeout, 120.0))
        async with httpx.AsyncClient(timeout=timeout) as http:
            snapshot = await self._submit_batch(http, payload, json_mode=json_mode, requests=requests)
            batch_id = snapshot.get("id")
            if not batch_id:
                raise TeacherError(f"batch submit missing id: {snapshot!r}"[:400])
            try:
                delay = max(0.05, poll_seconds)
                while snapshot.get("status") not in BATCH_TERMINAL:
                    if on_status:
                        await on_status(snapshot)
                    if check:
                        check()
                    await asyncio.sleep(delay)
                    snapshot = await self._poll_batch(http, str(batch_id))
                    delay = min(30.0, delay * 1.25)
                if on_status:
                    await on_status(snapshot)
            except BaseException:
                await _cancel_batch(http, f"{self.batch_url}/{batch_id}", self._headers())
                raise
        status = snapshot.get("status")
        if status != "completed":
            raise TeacherError(f"OpenRouter batch {batch_id} ended with status {status}")
        return _results_from_snapshot(requests, snapshot)

    async def _submit_batch(
        self,
        http: httpx.AsyncClient,
        payload: dict[str, Any],
        *,
        json_mode: bool,
        requests: list[BatchChatRequest],
    ) -> dict[str, Any]:
        try:
            response = await http.post(self.batch_url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise TeacherError(f"batch submit failed: {exc}") from exc
        if response.status_code >= 400 and json_mode:
            fallback = _batch_payload(self.batch_model, requests, json_mode=False)
            return await self._submit_batch(http, fallback, json_mode=False, requests=requests)
        if response.status_code >= 400:
            raise TeacherError(f"batch HTTP {response.status_code}: {response.text[:800]}")
        return _require_object(response, "batch submit")

    async def _poll_batch(self, http: httpx.AsyncClient, batch_id: str) -> dict[str, Any]:
        try:
            response = await http.get(f"{self.batch_url}/{batch_id}", headers=self._headers())
        except httpx.HTTPError as exc:
            raise TeacherError(f"batch poll failed: {exc}") from exc
        if response.status_code >= 400:
            raise TeacherError(f"batch poll HTTP {response.status_code}: {response.text[:800]}")
        return _require_object(response, "batch poll")


def _chat_body(model: str, system: str, user: str, temperature: float, json_mode: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _batch_payload(model: str, requests: list[BatchChatRequest], *, json_mode: bool) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for req in requests:
        body: dict[str, Any] = {
            "temperature": req.temperature,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        items.append({"custom_id": req.custom_id, "body": body})
    # endpoint + model must precede requests: OpenRouter stream-parses the body.
    return {
        "endpoint": "/v1/chat/completions",
        "model": model,
        "requests": items,
    }


def completion_content(data: Any) -> Any:
    try:
        message = data["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise TeacherError(f"unexpected teacher response: {data!r}"[:400]) from exc
    if isinstance(content, (dict, list)):
        return content
    if content in (None, ""):
        content = message.get("reasoning") or message.get("parsed") or ""
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return strip_think(str(content))


def coerce_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    try:
        return parse_json_payload(str(text))
    except TeacherError:
        preview = re.sub(r"\s+", " ", str(text))[:400]
        raise TeacherError(
            "teacher did not return valid JSON. "
            f"It sent JS/Python-like text instead of {{\"examples\":[...]}}. Preview: {preview}"
        ) from None


def content_as_text(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content)
    return str(content)


def _require_object(response: httpx.Response, label: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise TeacherError(f"{label} was not JSON: {response.text[:400]}") from exc
    if not isinstance(data, dict):
        raise TeacherError(f"unexpected {label}: {data!r}"[:400])
    return data


async def _cancel_batch(http: httpx.AsyncClient, url: str, headers: dict[str, str]) -> None:
    try:
        await http.post(f"{url}/cancel", headers=headers)
    except Exception:
        return


def _results_from_snapshot(requests: list[BatchChatRequest], snapshot: dict[str, Any]) -> list[BatchChatResult]:
    rows = snapshot.get("results") or []
    by_id = {item.get("custom_id"): item for item in rows if isinstance(item, dict)}
    out: list[BatchChatResult] = []
    for req in requests:
        item = by_id.get(req.custom_id)
        if not item:
            out.append(BatchChatResult(req.custom_id, None, "missing result"))
            continue
        error = item.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            out.append(BatchChatResult(req.custom_id, None, message or "batch item error"))
            continue
        response = item.get("response") or {}
        if not isinstance(response, dict):
            out.append(BatchChatResult(req.custom_id, None, "malformed batch response"))
            continue
        status_code = int(response.get("status_code") or 200)
        if status_code >= 400:
            body = response.get("body")
            out.append(BatchChatResult(req.custom_id, None, f"item HTTP {status_code}: {str(body)[:240]}"))
            continue
        try:
            content = completion_content(response.get("body") or {})
        except TeacherError as exc:
            out.append(BatchChatResult(req.custom_id, None, str(exc)))
            continue
        out.append(BatchChatResult(req.custom_id, content_as_text(content)))
    return out


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _looks_like_payload(obj: Any) -> bool:
    if isinstance(obj, dict) and ("examples" in obj or "items" in obj or "curriculum" in obj):
        return True
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return True
    return False


_UNQUOTED_KEY = re.compile(r"([{\[,]\s*)([A-Za-z_][\w]*)\s*:")


def _single_quotes_to_double(blob: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(blob):
        char = blob[index]
        if char != "'":
            out.append(char)
            index += 1
            continue
        index += 1
        chunk: list[str] = []
        while index < len(blob):
            if blob[index] == "\\" and index + 1 < len(blob):
                chunk.append(blob[index : index + 2])
                index += 2
                continue
            if blob[index] == "'":
                inner = "".join(chunk).replace('"', '\\"')
                out.append('"' + inner + '"')
                index += 1
                break
            chunk.append(blob[index])
            index += 1
        else:
            out.append("'")
            out.extend(chunk)
    return "".join(out)


def _normalize_jsonish(blob: str) -> str:
    blob = blob.strip().lstrip("\ufeff")
    blob = _UNQUOTED_KEY.sub(r'\1"\2":', blob)
    blob = _single_quotes_to_double(blob)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    blob = blob.replace("True", "true").replace("False", "false").replace("None", "null")
    return blob


def _loads_loose(blob: str) -> Any:
    attempts = [blob, _normalize_jsonish(blob)]
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            obj = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, MemoryError):
            obj = None
        if isinstance(obj, (dict, list)):
            return obj
    raise json.JSONDecodeError("loose parse failed", blob, 0)


def _jsonish_start(text: str, index: int) -> bool:
    nxt = text[index + 1 : index + 24].lstrip()
    return bool(nxt) and nxt[0] in "\"'{[tfn0123456789]}"


def _balanced_blob(text: str, start: int) -> str | None:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_payload(text: str) -> Any:
    text = text.strip()
    candidates = [text]
    fenced = JSON_FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    fallback: list[Any] = []
    last_err: Exception | None = None
    for candidate in candidates:
        try:
            loaded = _loads_loose(candidate)
        except json.JSONDecodeError as exc:
            last_err = exc
        else:
            return loaded
        for index, char in enumerate(candidate):
            if char not in "{[" or not _jsonish_start(candidate, index):
                continue
            blob = _balanced_blob(candidate, index) or candidate[index:]
            try:
                obj = _loads_loose(blob)
            except json.JSONDecodeError as exc:
                last_err = exc
                continue
            if isinstance(obj, dict) and ("examples" in obj or "items" in obj):
                return obj
            if _looks_like_payload(obj):
                fallback.append(obj)
    if fallback:
        return fallback[0]
    preview = re.sub(r"\s+", " ", text)[:240]
    detail = f"{last_err.msg} at pos {last_err.pos}" if isinstance(last_err, json.JSONDecodeError) else "no JSON object in reply"
    raise TeacherError(f"teacher did not return valid JSON ({detail}); got: {preview}")
