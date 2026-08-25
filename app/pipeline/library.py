from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from app.models import ShareGPTExample
from app.pipeline.jsonl import example_payload, read_jsonl, write_jsonl
from app.store import slugify

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def topic_slug(topic: str) -> str:
    raw = (topic or "general").strip()
    try:
        return slugify(raw)
    except ValueError:
        fallback = _NON_SLUG.sub("-", raw.lower()).strip("-")
        return fallback or "general"


def example_fingerprint(example: ShareGPTExample) -> str:
    existing = str(example.meta.get("fingerprint") or "").strip()
    if existing:
        return existing
    human = example.conversations[0].value.strip() if example.conversations else ""
    gpt = example.conversations[-1].value.strip() if len(example.conversations) > 1 else ""
    payload = f"{human}\n---\n{gpt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stamp_fingerprint(example: ShareGPTExample) -> ShareGPTExample:
    example.meta["fingerprint"] = example_fingerprint(example)
    return example


def group_by_topic(examples: list[ShareGPTExample]) -> dict[str, list[ShareGPTExample]]:
    groups: dict[str, list[ShareGPTExample]] = defaultdict(list)
    for example in examples:
        label = str(example.meta.get("topic") or "general")
        groups[label].append(example)
    return dict(groups)


def _fingerprints_on_disk(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for row in read_jsonl(path):
        seen.add(example_fingerprint(row))
    return seen


def append_unique(path: Path, examples: list[ShareGPTExample]) -> int:
    """Append unseen examples to a topic shard. Returns how many were new."""
    seen = _fingerprints_on_disk(path)
    fresh: list[ShareGPTExample] = []
    for example in examples:
        stamp_fingerprint(example)
        mark = example.meta["fingerprint"]
        if mark in seen:
            continue
        seen.add(mark)
        fresh.append(example)
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in fresh:
            fh.write(json.dumps(example_payload(row), ensure_ascii=False) + "\n")
    return len(fresh)


def write_topic_snapshot(path: Path, examples: list[ShareGPTExample]) -> None:
    unique: list[ShareGPTExample] = []
    seen: set[str] = set()
    for example in examples:
        stamp_fingerprint(example)
        mark = str(example.meta["fingerprint"])
        if mark in seen:
            continue
        seen.add(mark)
        unique.append(example)
    write_jsonl(path, unique)


def promote_examples(
    examples: list[ShareGPTExample],
    *,
    library_topic_jsonl,
    project_topic_jsonl,
) -> dict[str, int]:
    """Write this-run topic files and merge new rows into the reusable library."""
    added: dict[str, int] = {}
    for topic, rows in group_by_topic(examples).items():
        write_topic_snapshot(project_topic_jsonl(topic), rows)
        added[topic_slug(topic)] = append_unique(library_topic_jsonl(topic), rows)
    return added
