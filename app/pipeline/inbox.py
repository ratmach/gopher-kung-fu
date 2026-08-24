from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.models import ShareGPTExample, ShareGPTTurn
from app.pipeline.jsonl import example_payload


def example_from_payload(data: dict) -> ShareGPTExample:
    turns: list[ShareGPTTurn] = []
    for turn in data.get("conversations", []):
        role = turn.get("from") or turn.get("role")
        if role not in {"human", "gpt"}:
            continue
        turns.append(ShareGPTTurn(role=role, value=turn.get("value", "")))
    return ShareGPTExample(conversations=turns, meta=dict(data.get("meta") or {}))


def _item_prefix(item_id: str) -> str:
    return f"{item_id}__"


def example_key(item_id: str, slot: int) -> str:
    return f"{item_id}__{slot}"


class ExampleInbox:
    """Crash-safe keyed example store. Each kept example is flushed before the next teacher wait."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self.rows: dict[str, ShareGPTExample] = {}
        self._load()

    def _load(self) -> None:
        self.rows = {}
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(data.get("key") or "").strip()
            if not key:
                continue
            example = example_from_payload(data)
            example.meta["key"] = key
            self.rows[key] = example

    def count_for(self, item_id: str) -> int:
        prefix = _item_prefix(item_id)
        return sum(1 for key in self.rows if key.startswith(prefix))

    def examples_for(self, item_id: str) -> list[ShareGPTExample]:
        prefix = _item_prefix(item_id)
        keys = [key for key in self.rows if key.startswith(prefix)]

        def slot(key: str) -> int:
            try:
                return int(key[len(prefix) :])
            except ValueError:
                return 0

        keys.sort(key=slot)
        return [self.rows[key] for key in keys]

    def ordered(self, item_ids: list[str]) -> list[ShareGPTExample]:
        out: list[ShareGPTExample] = []
        for item_id in item_ids:
            out.extend(self.examples_for(item_id))
        return out

    def clear(self) -> None:
        self.rows = {}
        if self.path.exists():
            self.path.unlink()

    async def put(self, item_id: str, example: ShareGPTExample, *, limit: int) -> str | None:
        async with self._lock:
            if self.count_for(item_id) >= limit:
                return None
            key = example_key(item_id, self.count_for(item_id))
            example.meta["key"] = key
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"key": key, **example_payload(example)}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                fh.flush()
            self.rows[key] = example
            return key
