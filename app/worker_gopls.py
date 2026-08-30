from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INIT_TIMEOUT = 25.0
REQUEST_TIMEOUT = 12.0
DIAG_TIMEOUT = 15.0


@dataclass
class GoplsDiagnostic:
    rel: str
    message: str
    line: int
    character: int
    end_line: int
    end_character: int
    source: str = ""


class GoplsError(RuntimeError):
    pass


class GoplsSession:
    """One gopls stdio LSP session for a Go module root."""

    def __init__(self, root: Path, exe: str) -> None:
        self.root = root.resolve()
        self.exe = exe
        self._proc: subprocess.Popen[bytes] | None = None
        self._id = 0
        self._pending: dict[int, Future] = {}
        self._diags: dict[str, list[dict[str, Any]]] = {}
        self._diag_events: dict[str, threading.Event] = {}
        self._versions: dict[str, int] = {}
        self._lock = threading.Lock()
        self._alive = False
        self._stderr_thread: threading.Thread | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [self.exe, "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.root),
            bufsize=0,
        )
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, name="gopls-lsp", daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="gopls-err", daemon=True)
        self._stderr_thread.start()
        root_uri = self.root.as_uri()
        result = self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
                "capabilities": {
                    "textDocument": {
                        "completion": {"completionItem": {"snippetSupport": False}},
                        "publishDiagnostics": {"relatedInformation": False},
                        "synchronization": {"didSave": True},
                    },
                    "workspace": {"workspaceFolders": True},
                },
            },
            timeout=INIT_TIMEOUT,
        )
        if not isinstance(result, dict):
            raise GoplsError("gopls initialize failed")
        self._notify("initialized", {})

    def close(self) -> None:
        self._alive = False
        proc = self._proc
        if proc is None:
            return
        try:
            self._notify("exit", None)
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._proc = None

    def sync_file(self, rel: str) -> None:
        path = (self.root / rel).resolve()
        if not path.is_file():
            return
        text = path.read_text(encoding="utf-8")
        uri = path.as_uri()
        with self._lock:
            version = self._versions.get(uri, 0) + 1
            self._versions[uri] = version
            event = self._diag_events.setdefault(uri, threading.Event())
            event.clear()
        if version == 1:
            self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "go",
                        "version": version,
                        "text": text,
                    }
                },
            )
        else:
            self._notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        event.wait(timeout=DIAG_TIMEOUT)

    def diagnostics_for(self, rel: str) -> list[GoplsDiagnostic]:
        path = (self.root / rel).resolve()
        uri = path.as_uri()
        with self._lock:
            raw = list(self._diags.get(uri) or [])
        out: list[GoplsDiagnostic] = []
        for item in raw:
            rng = item.get("range") or {}
            start = rng.get("start") or {}
            end = rng.get("end") or {}
            out.append(
                GoplsDiagnostic(
                    rel=rel.replace("\\", "/"),
                    message=str(item.get("message") or ""),
                    line=int(start.get("line") or 0),
                    character=int(start.get("character") or 0),
                    end_line=int(end.get("line") or start.get("line") or 0),
                    end_character=int(end.get("character") or start.get("character") or 0),
                    source=str(item.get("source") or ""),
                )
            )
        return out

    def completions_at(self, rel: str, line: int, character: int) -> list[str]:
        path = (self.root / rel).resolve()
        uri = path.as_uri()
        try:
            result = self._request(
                "textDocument/completion",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": int(line), "character": int(character)},
                },
                timeout=REQUEST_TIMEOUT,
            )
        except GoplsError:
            return []
        items: list[dict[str, Any]]
        if isinstance(result, list):
            items = [x for x in result if isinstance(x, dict)]
        elif isinstance(result, dict):
            items = [x for x in (result.get("items") or []) if isinstance(x, dict)]
        else:
            items = []
        names: list[str] = []
        seen: set[str] = set()
        for item in items:
            kind = item.get("kind")
            if kind in {14, 15}:  # Keyword, Snippet
                continue
            label = str(item.get("insertText") or item.get("label") or "").strip()
            label = label.split("(")[0].split(" ")[0].strip()
            if not label or not label[:1].isalpha() or label in seen:
                continue
            seen.add(label)
            names.append(label)
            if len(names) >= 20:
                break
        return names

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while self._alive and proc.stderr.read(1024):
                pass
        except Exception:
            return

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        while self._alive:
            msg = _read_lsp(stream)
            if msg is None:
                break
            method = msg.get("method")
            if method == "textDocument/publishDiagnostics":
                params = msg.get("params") or {}
                uri = str(params.get("uri") or "")
                diags = params.get("diagnostics") or []
                with self._lock:
                    self._diags[uri] = list(diags) if isinstance(diags, list) else []
                    event = self._diag_events.get(uri)
                    if event is not None:
                        event.set()
                continue
            if method and msg.get("id") is not None:
                self._reply(msg["id"], None)
                continue
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            fut = self._pending.get(int(msg_id))
            if fut is None or fut.done():
                continue
            if "error" in msg:
                fut.set_exception(GoplsError(str(msg.get("error"))))
            else:
                fut.set_result(msg.get("result"))

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _request(self, method: str, params: Any, *, timeout: float) -> Any:
        msg_id = self._next_id()
        fut: Future = Future()
        self._pending[msg_id] = fut
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            raise GoplsError(f"{method}: {exc}") from exc
        finally:
            self._pending.pop(msg_id, None)

    def _notify(self, method: str, params: Any) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def _reply(self, msg_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise GoplsError("gopls is not running")
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        with self._lock:
            proc.stdin.write(header + raw)
            proc.stdin.flush()


def _read_lsp(stream) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError:
            continue
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    try:
        size = int(headers.get("content-length") or "0")
    except ValueError:
        return None
    body = b""
    while len(body) < size:
        chunk = stream.read(size - len(body))
        if not chunk:
            return None
        body += chunk
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def start_gopls(root: Path | None) -> GoplsSession | None:
    if root is None:
        return None
    flag = os.environ.get("GOPHER_GOPLS", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return None
    exe = shutil.which("gopls")
    if not exe:
        return None
    try:
        session = GoplsSession(root, exe)
        session.start()
        return session
    except Exception:
        return None


def missing_method_from_message(message: str) -> str:
    text = message or ""
    marker = "has no field or method "
    idx = text.lower().rfind(marker)
    if idx >= 0:
        rest = text[idx + len(marker) :].strip()
        name = rest.split()[0].strip(").,") if rest else ""
        return name
    return ""


def type_from_missing_message(message: str) -> str:
    text = message or ""
    lower = text.lower()
    start = lower.find("(type ")
    end = lower.find(" has no field")
    if start >= 0 and end > start:
        return text[start + 6 : end].strip()
    return ""


def unused_name_from_message(message: str) -> str:
    text = message or ""
    marker = "declared and not used:"
    idx = text.lower().rfind(marker)
    if idx < 0:
        return ""
    rest = text[idx + len(marker) :].strip()
    name = rest.split()[0].strip(").,") if rest else ""
    return name if name.isidentifier() else ""
