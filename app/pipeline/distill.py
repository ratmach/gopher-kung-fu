from __future__ import annotations

import asyncio
import math
from collections import defaultdict

import httpx

from app.jobs import JobCancelled, JobHub
from app.models import Curriculum, CurriculumItem, DistillSettings, Project, ShareGPTExample, ShareGPTTurn
from app.pipeline.filters import keep_example
from app.pipeline.inbox import ExampleInbox
from app.pipeline.jsonl import split_train_eval, write_jsonl
from app.store import ProjectStore
from app.teachers.client import BatchChatRequest, TeacherClient, TeacherError, coerce_json

SYSTEM = """You write synthetic supervised fine-tuning data for a specialist coding SLM.
Produce realistic developer tasks. The assistant answer must be the specialist: concrete, correct, and terse.
Always include at least one fenced code block in the assistant reply.
Do not include <think> traces, chain-of-thought, or preamble about being an AI.
Return JSON only: {"examples":[{"human":"...","gpt":"..."}]}
Each example is one user request and one assistant answer.
Skills:
- write: implement from a spec
- review: find bugs or smells in a snippet, then show a fix
- debug: broken code plus a traceback or symptom, then the fix
- refactor: improve structure without changing behavior
- idiom: the right library/API usage for this niche
"""

CODE_SKILLS = {"write", "review", "debug", "refactor", "idiom"}
REDO_STATUSES = {"ready_to_train", "trained", "exporting", "exported"}


def _batch_size(remaining: int) -> int:
    return max(1, min(4, remaining))


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
    return (
        f"Specialist: {project.name}\n"
        f"Niche topics: {', '.join(t.label for t in project.topics)}\n"
        f"Syllabus item: {item.topic} — {item.subtopic}\n"
        f"Skill: {item.skill}\n"
        f"Difficulty: {item.difficulty}\n"
        f"Notes: {item.notes or 'none'}\n"
        f"Write {count} diverse examples. Different APIs, bugs, or constraints. No duplicated prompts.\n"
        "User messages should look like a developer request. Assistant messages should solve it with code."
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
    return examples


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
    if project.status in REDO_STATUSES:
        inbox.clear()
        inbox = ExampleInbox(store.inbox_jsonl(project.slug))
    elif inbox.rows:
        await hub.log(job_id, f"Resuming {len(inbox.rows)} saved example(s) from synthetic/inbox.jsonl.")

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
    await hub.log(
        job_id,
        f"Wrote {len(train_rows)} train and {len(eval_rows)} eval examples. Dropped empties / no-code replies.",
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
    hub.check(job_id)
    remaining = quota - inbox.count_for(item.id)
    if remaining <= 0:
        return 0
    examples = await _generate_batch(client, project, item, remaining, http=http)
    return await _record_examples(inbox, item, examples, quota, hub, job_id)


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
