from __future__ import annotations

import json
from pathlib import Path

from app.models import ShareGPTExample, ShareGPTTurn


def example_payload(row: ShareGPTExample) -> dict:
    return {
        "conversations": [turn.to_unsloth() for turn in row.conversations],
        "meta": row.meta,
    }


def write_jsonl(path: Path, rows: list[ShareGPTExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(example_payload(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[ShareGPTExample]:
    if not path.exists():
        return []
    rows: list[ShareGPTExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        turns = []
        for turn in data.get("conversations", []):
            role = turn.get("from") or turn.get("role")
            if role not in {"human", "gpt"}:
                continue
            turns.append(ShareGPTTurn(role=role, value=turn.get("value", "")))
        rows.append(ShareGPTExample(conversations=turns, meta=data.get("meta", {})))
    return rows


def split_train_eval(
    rows: list[ShareGPTExample],
    eval_ratio: float = 0.1,
) -> tuple[list[ShareGPTExample], list[ShareGPTExample]]:
    if len(rows) < 10:
        return rows, []
    eval_n = max(1, int(len(rows) * eval_ratio))
    return rows[eval_n:], rows[:eval_n]
