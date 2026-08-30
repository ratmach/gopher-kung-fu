from __future__ import annotations

import asyncio
import math
from collections import defaultdict

import httpx

from app.jobs import JobCancelled, JobHub
from app.models import Curriculum, CurriculumItem, DistillSettings, Project, ShareGPTExample, ShareGPTTurn
from app.pipeline.compile_gate import compile_gate_example
from app.pipeline.filters import keep_example
from app.pipeline.inbox import ExampleInbox
from app.pipeline.jsonl import split_train_eval, write_jsonl
from app.pipeline.library import example_fingerprint, promote_examples, stamp_fingerprint
from app.pipeline.seeds import (
    GC_CONVERT_ZERO,
    GC_MAP_REASSIGN,
    gold_examples,
    is_go_language_topic,
)
from app.store import ProjectStore
from app.teachers.client import BatchChatRequest, TeacherClient, TeacherError, coerce_json

SYSTEM = """You write synthetic supervised fine-tuning data for a specialist coding SLM.
Produce realistic developer tasks. The assistant answer must be the specialist: concrete and almost entirely code.
Always include at least one fenced code block in the assistant reply.
Do not include <think> traces, chain-of-thought, or preamble about being an AI.
Return JSON only: {"examples":[{"human":"...","gpt":"..."}]}
Each example is one user request and one assistant answer.

Human message must look like a worker packet, not an essay. Include:
- what to build (behavior + tests that must pass)
- constraints (allowed modules, no-CGO, layout, names)
- the exact relative .go path to write
- neighboring signatures when the impl must honor them

Assistant message (gpt):
- Almost only fenced code. At most two sentences after the last fence. No architecture lectures, package shopping lists, or "the rest is similar".
- write: one complete compiling unit. Prefer this shape:
  ### relative/path.go
  ```go
  package ...
  ```
- debug / repair: the human contains the broken unit plus a verbatim `gc` or `go test` line (copy the compiler, including Python-isms it rejects: for/else, elif, try/except; type errors: cannot convert 0 (untyped int constant) to type interface{Error() string} from `return 0` / `err == 0` / `WriteByte(...) == 0` — gpt uses `return nil` / `err != nil` / `if err := w.WriteByte(c); err != nil`; `w.Write(r)` with r rune — gc cannot use r (variable of type rune) as []byte — gpt uses `w.Write([]byte(string(r)))` or `io.WriteString(w, string(r))`, not WriteRune on io.Writer; `seen := map[string]int{}` then assigned `map[string][]string` — gpt declares the map with the value type it stores; and `declared and not used: x` — gpt USES named locals (append, increment, return them); `_` only in `for _, v := range`, never `_ = buf`). If the host attached a member list after the error, gpt must pick from those names, not invent APIs. gpt is the patched complete file in the same ### path.go fence. Do not echo the error inside a fence. Do not write a language-comparison essay.
- review / refactor / idiom: still a corrected or canonical code block, not a bullet-list review. Prefer write and debug; these other skills are secondary.
"""

CODE_SKILLS = {"write", "review", "debug", "refactor", "idiom"}
REDO_STATUSES = {"ready_to_train", "trained", "exporting", "exported"}


def _batch_size(remaining: int) -> int:
    return max(1, min(4, remaining))


def _is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "401" in text or "403" in text or "user not found" in text or "no api key" in text


def _distill_settings(project: Project) -> DistillSettings:
    raw = project.distill
    if isinstance(raw, DistillSettings):
        return raw
    return DistillSettings.model_validate(raw)


def _quota_per_item(project: Project, curriculum: Curriculum) -> dict[str, int]:
    if not curriculum.items:
        return {}
    by_topic: dict[str, list[str]] = defaultdict(list)
    for item in curriculum.items:
        by_topic[item.topic].append(item.id)
    target = max(8, _distill_settings(project).examples_per_topic)
    quotas: dict[str, int] = {}
    for ids in by_topic.values():
        base, extra = divmod(target, len(ids))
        for index, item_id in enumerate(ids):
            quotas[item_id] = max(1, base + (1 if index < extra else 0))
    for item in curriculum.items:
        q = quotas.get(item.id, 1)
        if item.skill not in {"write", "debug"}:
            quotas[item.id] = max(1, q // 2)
    return quotas


def _items_by_topic(curriculum: Curriculum) -> list[tuple[str, list[CurriculumItem]]]:
    groups: dict[str, list[CurriculumItem]] = {}
    order: list[str] = []
    for item in curriculum.items:
        if item.topic not in groups:
            order.append(item.topic)
            groups[item.topic] = []
        groups[item.topic].append(item)
    return [(topic, groups[topic]) for topic in order]


def _user_prompt(project: Project, item: CurriculumItem, count: int) -> str:
    skill_hint = (
        "Human: short spec + constraints + file path + signatures. "
        "gpt: complete ### path.go fence that would compile."
    )
    if item.skill == "debug":
        skill_hint = (
            "Human: REPAIR packet — previous broken ### path.go file plus a verbatim "
            "`gc`/`go test` line (filename:line:col: message), not a paraphrase. "
            "Prefer illegal Go `gc` actually emits: unexpected keyword else (for/else), "
            "elif, try/except, def, declared and not used, undefined method, "
            f"and {GC_CONVERT_ZERO} from `return 0` / `err == 0` / `WriteByte(...) == 0` "
            "(gpt: `return nil` / `err != nil` / `if err := w.WriteByte(c); err != nil`), "
            "`w.Write(r)` rune (gpt: `w.Write([]byte(string(r)))` / `io.WriteString`, "
            "not WriteRune on io.Writer), "
            f"`seen := map[string]int{{}}` then assigned map[string][]string "
            f"(gc: {GC_MAP_REASSIGN}; gpt: declare `seen := map[string][]string{{}}`), "
            "and declared and not used: x (gpt: use x — append/increment/return it; "
            "`_` only in `for _, v := range`, never `_ = buf`). "
            "When the packet includes a member list, gpt picks from it. gpt: patched complete "
            "compiling ### path.go only. No Python-vs-Go lecture."
        )
    elif item.skill not in {"write", "debug"}:
        skill_hint = (
            "Still return a complete canonical code fence, not a review essay. "
            "Keep this secondary to write/debug work."
        )
    return (
        f"Specialist: {project.name}\n"
        f"Niche topics: {', '.join(t.label for t in project.topics)}\n"
        f"Syllabus item: {item.topic} — {item.subtopic}\n"
        f"Skill: {item.skill}\n"
        f"Difficulty: {item.difficulty}\n"
        f"Notes: {item.notes or 'none'}\n"
        f"Write {count} diverse examples. Different APIs, bugs, or constraints. No duplicated prompts.\n"
        f"{skill_hint}\n"
        "Human messages are worker packets (spec, constraints, files, existing signatures).\n"
        "Assistant messages are complete fenced code — full file — not sketches or advice.\n"
        "If this item is write or debug, the gpt field must be dominated by code fences."
    )


def _examples_from_raw(raw: object, item: CurriculumItem, count: int) -> list[ShareGPTExample]:
    rows = raw.get("examples") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise TeacherError("distill JSON must contain an examples array")
    examples: list[ShareGPTExample] = []
    for row in rows[: count + 1]:
        if not isinstance(row, dict):
            continue
        human = str(row.get("human") or row.get("user") or "").strip()
        gpt = str(row.get("gpt") or row.get("assistant") or "").strip()
        if not human or not gpt:
            continue
        examples.append(
            ShareGPTExample(
                conversations=[
                    ShareGPTTurn(role="human", value=human),
                    ShareGPTTurn(role="gpt", value=gpt),
                ],
                meta={
                    "topic": item.topic,
                    "subtopic": item.subtopic,
                    "skill": item.skill,
                    "difficulty": item.difficulty,
                    "item_id": item.id,
                },
            )
        )
    for example in examples:
        stamp_fingerprint(example)
    return examples


def _short_items(inbox: ExampleInbox, curriculum: Curriculum, quotas: dict[str, int]) -> list[CurriculumItem]:
    return [item for item in curriculum.items if inbox.count_for(item.id) < quotas.get(item.id, 1)]


def _project_has_go(project: Project, curriculum: Curriculum) -> bool:
    if any(is_go_language_topic(t.id, t.label) for t in project.topics):
        return True
    return any(is_go_language_topic("", item.topic) for item in curriculum.items)


def _item_for_seed(curriculum: Curriculum, base_id: str) -> CurriculumItem | None:
    for item in curriculum.items:
        if item.id == base_id or item.id.startswith(f"{base_id}-"):
            return item
    return None


def _merge_gold(kept: list[ShareGPTExample], project: Project, curriculum: Curriculum) -> list[ShareGPTExample]:
    if not _project_has_go(project, curriculum):
        return kept
    seen = {example_fingerprint(example) for example in kept}
    extra: list[ShareGPTExample] = []
    for gold in gold_examples():
        stamp_fingerprint(gold)
        mark = str(gold.meta["fingerprint"])
        if mark in seen:
            continue
        if not keep_example(gold):
            continue
        ok, _reason = compile_gate_example(gold)
        if not ok:
            continue
        extra.append(gold)
        seen.add(mark)
    return extra + kept


async def _seed_compiler_gold(
    inbox: ExampleInbox,
    curriculum: Curriculum,
    quotas: dict[str, int],
    hub: JobHub,
    job_id: str,
) -> int:
    added = 0
    for gold in gold_examples():
        item_id = str(gold.meta.get("item_id") or "")
        item = _item_for_seed(curriculum, item_id)
        if item is None:
            await hub.log(job_id, f"Skipped gold seed {item_id}: no matching syllabus item")
            continue
        gold.meta["topic"] = item.topic
        gold.meta["subtopic"] = item.subtopic
        gold.meta["item_id"] = item.id
        added += await _record_examples(inbox, item, [gold], quotas.get(item.id, 1), hub, job_id)
    return added


async def distill_project(
    project: Project,
    curriculum: Curriculum,
    client: TeacherClient,
    store: ProjectStore,
    hub: JobHub,
    job_id: str,
) -> tuple[int, int]:
    settings = _distill_settings(project)
    inbox = ExampleInbox(store.inbox_jsonl(project.slug))
    quotas = _quota_per_item(project, curriculum)
    short = _short_items(inbox, curriculum, quotas)
    if project.status in REDO_STATUSES and not short:
        inbox.clear()
        inbox = ExampleInbox(store.inbox_jsonl(project.slug))
        await hub.log(job_id, "Inbox already full for this syllabus — starting a fresh distill.")
    elif inbox.rows:
        await hub.log(job_id, f"Resuming {len(inbox.rows)} saved example(s) from synthetic/inbox.jsonl.")

    seeded = await _seed_compiler_gold(inbox, curriculum, quotas, hub, job_id)
    if seeded:
        await hub.log(job_id, f"Seeded {seeded} Go compiler gold row(s).")

    if settings.use_batch:
        if not client.supports_batch:
            raise TeacherError(
                "OpenRouter batch is on, but this teacher is not OpenRouter. "
                "Turn batch off on the Distill step, or switch the teacher to OpenRouter."
            )
        failures = await _distill_via_batch(project, curriculum, client, hub, job_id, inbox)
    else:
        failures = await _distill_live(project, curriculum, client, hub, job_id, inbox)
    kept = inbox.ordered([item.id for item in curriculum.items])
    kept = _merge_gold(kept, project, curriculum)
    return await _write_examples(project, store, hub, job_id, kept, failures)


async def _write_examples(
    project: Project,
    store: ProjectStore,
    hub: JobHub,
    job_id: str,
    kept: list[ShareGPTExample],
    failures: int,
) -> tuple[int, int]:
    if len(kept) < 4:
        raise TeacherError(f"not enough usable examples after filters ({len(kept)})")
    train_rows, eval_rows = split_train_eval(kept)
    write_jsonl(store.train_jsonl(project.slug), train_rows)
    write_jsonl(store.eval_jsonl(project.slug), eval_rows)
    promoted = promote_examples(
        kept,
        library_topic_jsonl=store.library_topic_jsonl,
        project_topic_jsonl=lambda topic: store.project_topic_jsonl(project.slug, topic),
    )
    shard_bits = ", ".join(f"{name}+{count}" for name, count in promoted.items()) or "none"
    await hub.log(
        job_id,
        f"Wrote {len(train_rows)} train and {len(eval_rows)} eval examples. "
        f"Topic shards updated ({shard_bits}). Dropped empties / essay-heavy replies.",
        1.0,
    )
    if failures:
        await hub.log(job_id, f"Completed with {failures} teacher batch error(s).")
    return len(train_rows), len(eval_rows)


async def _record_examples(
    inbox: ExampleInbox,
    item: CurriculumItem,
    examples: list[ShareGPTExample],
    quota: int,
    hub: JobHub,
    job_id: str,
) -> int:
    added = 0
    for example in examples:
        if not keep_example(example):
            continue
        ok, reason = compile_gate_example(example)
        if not ok:
            await hub.log(job_id, f"Dropped {item.subtopic} (compile-gate: {reason[:120]})")
            continue
        key = await inbox.put(item.id, example, limit=quota)
        if key is None:
            break
        added += 1
        have = inbox.count_for(item.id)
        await hub.log(job_id, f"Saved {key} — {item.subtopic} {have}/{quota}")
    return added


def _progress(inbox: ExampleInbox, total_target: int, topic_index: int, topic_count: int) -> float:
    saved = len(inbox.rows) / max(1, total_target)
    topics = topic_index / max(1, topic_count)
    return 0.05 + 0.85 * min(1.0, max(saved, topics))


async def _distill_live(
    project: Project,
    curriculum: Curriculum,
    client: TeacherClient,
    hub: JobHub,
    job_id: str,
    inbox: ExampleInbox,
) -> int:
    quotas = _quota_per_item(project, curriculum)
    total_target = sum(quotas.values())
    groups = _items_by_topic(curriculum)
    await hub.log(
        job_id,
        f"Distilling ~{total_target} coding examples across {len(curriculum.items)} items in {len(groups)} topics. "
        "One topic at a time; all items in a topic run in parallel. Each kept example is saved immediately.",
        0.02,
    )
    failures = 0
    timeout = getattr(client, "timeout", 120.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        for topic_index, (topic, items) in enumerate(groups):
            hub.check(job_id)
            topic_need = sum(quotas.get(item.id, 1) for item in items)
            already = sum(inbox.count_for(item.id) for item in items)
            await hub.log(
                job_id,
                f"Topic {topic}: {len(items)} items, {already}/{topic_need} saved. Filling the rest in parallel.",
                _progress(inbox, total_target, topic_index, len(groups)),
            )
            for wave in range(4):
                hub.check(job_id)
                todo = [item for item in items if inbox.count_for(item.id) < quotas.get(item.id, 1)]
                if not todo:
                    break
                if wave:
                    have = sum(inbox.count_for(item.id) for item in items)
                    await hub.log(job_id, f"{topic}: {have}/{topic_need}, retrying short items in parallel…")
                results = await asyncio.gather(
                    *[_fill_item_live(client, project, item, quotas.get(item.id, 1), inbox, hub, job_id, http) for item in todo],
                    return_exceptions=True,
                )
                for item, result in zip(todo, results, strict=True):
                    if isinstance(result, (JobCancelled, asyncio.CancelledError)):
                        raise result
                    if isinstance(result, BaseException):
                        failures += 1
                        await hub.log(job_id, f"Teacher error on {item.id}: {result}")
                        if _is_auth_error(result):
                            raise TeacherError(str(result)) from result
            have = sum(inbox.count_for(item.id) for item in items)
            await hub.log(
                job_id,
                f"[{topic_index + 1}/{len(groups)}] {topic} → {have}/{topic_need} examples",
                _progress(inbox, total_target, topic_index + 1, len(groups)),
            )
    return failures


async def _fill_item_live(
    client: TeacherClient,
    project: Project,
    item: CurriculumItem,
    quota: int,
    inbox: ExampleInbox,
    hub: JobHub,
    job_id: str,
    http: httpx.AsyncClient,
) -> int:
    added = 0
    while True:
        hub.check(job_id)
        remaining = quota - inbox.count_for(item.id)
        if remaining <= 0:
            return added
        have = inbox.count_for(item.id)
        await hub.log(
            job_id,
            f"Asking teacher for {remaining} on {item.subtopic} ({have}/{quota})",
        )
        examples = await _generate_batch(client, project, item, remaining, http=http)
        got = await _record_examples(inbox, item, examples, quota, hub, job_id)
        added += got
        if got == 0:
            return added
    return added


async def _distill_via_batch(
    project: Project,
    curriculum: Curriculum,
    client: TeacherClient,
    hub: JobHub,
    job_id: str,
    inbox: ExampleInbox,
) -> int:
    quotas = _quota_per_item(project, curriculum)
    total_target = sum(quotas.values())
    groups = _items_by_topic(curriculum)
    await hub.log(
        job_id,
        f"OpenRouter batch distill: ~{total_target} examples across {len(groups)} topics. "
        "One topic per batch. Each kept example is saved immediately. Leave the factory running.",
        0.02,
    )
    failures = 0
    last_status: tuple | None = None

    async def on_status(snapshot: dict) -> None:
        nonlocal last_status
        counts = snapshot.get("request_counts") or {}
        key = (snapshot.get("status"), counts.get("completed"), counts.get("failed"))
        if key == last_status:
            return
        last_status = key
        cost = (snapshot.get("usage") or {}).get("cost")
        extra = f" · billed ${cost}" if isinstance(cost, (int, float)) else ""
        await hub.log(
            job_id,
            f"OpenRouter batch {snapshot.get('id')}: {snapshot.get('status')} "
            f"{counts.get('completed', 0)}/{counts.get('total', 0)} completed{extra}",
            _progress(inbox, total_target, 0, max(1, len(groups))),
        )

    for topic_index, (topic, items) in enumerate(groups):
        hub.check(job_id)
        topic_need = sum(quotas.get(item.id, 1) for item in items)
        already = sum(inbox.count_for(item.id) for item in items)
        await hub.log(
            job_id,
            f"Topic {topic}: {already}/{topic_need} saved. Submitting the rest as one OpenRouter batch.",
            _progress(inbox, total_target, topic_index, len(groups)),
        )
        for wave in range(4):
            hub.check(job_id)
            filled = {item.id: inbox.examples_for(item.id) for item in items}
            planned = _plan_wave(project, items, quotas, filled, wave)
            if not planned:
                break
            last_status = None
            await hub.log(job_id, f"{topic} batch wave {wave + 1}: {len(planned)} request(s).")
            results = await client.run_chat_batch(
                [
                    BatchChatRequest(custom_id=custom_id, system=SYSTEM, user=prompt, temperature=0.7)
                    for custom_id, _item, _count, prompt in planned
                ],
                json_mode=True,
                check=lambda: hub.check(job_id),
                on_status=on_status,
            )
            lookup = {custom_id: (item, count) for custom_id, item, count, _prompt in planned}
            for result in results:
                item, count = lookup[result.custom_id]
                if result.error or not result.text:
                    failures += 1
                    await hub.log(job_id, f"Batch item {item.id} failed: {result.error or 'empty'}")
                    continue
                try:
                    examples = _examples_from_raw(coerce_json(result.text), item, count)
                except TeacherError as exc:
                    failures += 1
                    await hub.log(job_id, f"Batch item {item.id} was not usable JSON: {exc}")
                    continue
                await _record_examples(inbox, item, examples, quotas.get(item.id, 1), hub, job_id)
        have = sum(inbox.count_for(item.id) for item in items)
        await hub.log(
            job_id,
            f"[{topic_index + 1}/{len(groups)}] {topic} → {have}/{topic_need} examples",
            _progress(inbox, total_target, topic_index + 1, len(groups)),
        )
    return failures


def _plan_wave(
    project: Project,
    items: list[CurriculumItem],
    quotas: dict[str, int],
    produced: dict[str, list[ShareGPTExample]],
    wave: int,
) -> list[tuple[str, CurriculumItem, int, str]]:
    planned: list[tuple[str, CurriculumItem, int, str]] = []
    for item in items:
        remaining = quotas.get(item.id, 1) - len(produced.get(item.id) or [])
        seq = 0
        while remaining > 0:
            count = _batch_size(remaining)
            remaining -= count
            seq += 1
            custom_id = f"{item.id}__w{wave}__{seq}"
            planned.append((custom_id, item, count, _user_prompt(project, item, count)))
    return planned


async def _generate_batch(
    client: TeacherClient,
    project: Project,
    item: CurriculumItem,
    count: int,
    http: httpx.AsyncClient | None = None,
) -> list[ShareGPTExample]:
    raw = await client.chat_json(
        SYSTEM, _user_prompt(project, item, count), temperature=0.7, http=http
    )
    return _examples_from_raw(raw, item, count)


def planned_count(project: Project, curriculum: Curriculum) -> int:
    quotas = _quota_per_item(project, curriculum)
    return int(math.fsum(quotas.values())) if quotas else 0
