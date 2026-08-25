from __future__ import annotations

from app.jobs import JobHub
from app.models import Curriculum, CurriculumItem, Project, TopicRef, utcnow
from app.teachers.client import TeacherClient, TeacherError

SKILLS = ["write", "review", "debug", "refactor", "idiom"]
MIN_ITEMS_PER_TOPIC = 4
MAX_ITEMS_PER_TOPIC = 80
DEFAULT_ITEMS_PER_TOPIC = 12
TEACHER_BATCH = 16


def clamp_items_per_topic(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n = DEFAULT_ITEMS_PER_TOPIC
    return max(MIN_ITEMS_PER_TOPIC, min(MAX_ITEMS_PER_TOPIC, n))


def _system(count: int, topic_label: str) -> str:
    return f"""You design a narrow coding curriculum for a specialist small language model.
The model will be fine-tuned only on this niche. Stay specific. Do not broaden into unrelated stacks.
Return JSON only: {{"items":[{{"id":"kebab-id","topic":"{topic_label}","subtopic":"...","skill":"write|review|debug|refactor|idiom","difficulty":"easy|medium|hard","notes":"what this item should teach"}}]}}
Cover only the topic "{topic_label}". Mix the five skills. Prefer concrete library/API/idiom work over essays.
Return exactly {count} items. Each subtopic must be distinct.
"""


async def generate_curriculum(
    project: Project,
    client: TeacherClient,
    hub: JobHub,
    job_id: str,
) -> Curriculum:
    hub.check(job_id)
    n = clamp_items_per_topic(getattr(project, "items_per_topic", DEFAULT_ITEMS_PER_TOPIC))
    topics = list(project.topics)
    if not topics:
        raise TeacherError("select at least one topic")
    total = n * len(topics)
    await hub.log(
        job_id,
        f"Asking teacher for {n} syllabus items × {len(topics)} topics ({total} total)…",
        0.05,
    )
    items: list[CurriculumItem] = []
    seen: set[str] = set()
    for index, topic in enumerate(topics):
        hub.check(job_id)
        chunk = await _fill_topic(project, topic, n, client, hub, job_id, seen)
        items.extend(chunk)
        progress = 0.1 + 0.85 * (index + 1) / len(topics)
        await hub.log(job_id, f"{topic.label}: {len(chunk)}/{n} items.", progress)
    if not items:
        raise TeacherError("teacher returned an empty curriculum")
    curriculum = Curriculum(items=items, generated_at=utcnow())
    await hub.log(job_id, f"Syllabus ready: {len(items)} items.", 1.0)
    return curriculum


async def _fill_topic(
    project: Project,
    topic: TopicRef,
    n: int,
    client: TeacherClient,
    hub: JobHub,
    job_id: str,
    seen: set[str],
) -> list[CurriculumItem]:
    items: list[CurriculumItem] = []
    empty_rounds = 0
    while len(items) < n:
        hub.check(job_id)
        need = min(TEACHER_BATCH, n - len(items))
        covered = ", ".join(item.subtopic for item in items[-12:]) or "(none yet)"
        user = (
            f"Specialist name: {project.name} ({project.slug})\n"
            f"Topic: {topic.label} ({topic.id})\n"
            f"Return exactly {need} new syllabus items for this topic only.\n"
            f"Progress: {len(items)}/{n} already written. Do not repeat: {covered}"
        )
        raw = await client.chat_json(_system(need, topic.label), user, temperature=0.3)
        chunk = _coerce_items(raw, seen)
        for item in chunk:
            item.topic = topic.label
        chunk = chunk[:need]
        if not chunk:
            empty_rounds += 1
            if empty_rounds >= 2:
                break
            continue
        empty_rounds = 0
        items.extend(chunk)
        if need == TEACHER_BATCH and len(items) < n:
            await hub.log(
                job_id,
                f"{topic.label}: {len(items)}/{n}…",
            )
    return items[:n]


def _coerce_items(raw: object, seen: set[str] | None = None) -> list[CurriculumItem]:
    payload = raw
    if isinstance(raw, dict):
        payload = raw.get("items") or raw.get("curriculum") or []
    if not isinstance(payload, list):
        raise TeacherError("curriculum JSON must contain an items array")
    items: list[CurriculumItem] = []
    used = seen if seen is not None else set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        skill = str(row.get("skill", "write")).lower()
        if skill not in SKILLS:
            skill = "write"
        difficulty = str(row.get("difficulty", "medium")).lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        ident = str(row.get("id") or f"item-{index+1}")
        ident = ident.strip().lower().replace(" ", "-")
        if ident in used:
            ident = f"{ident}-{index+1}"
        used.add(ident)
        items.append(
            CurriculumItem(
                id=ident,
                topic=str(row.get("topic") or "general").strip(),
                subtopic=str(row.get("subtopic") or row.get("title") or ident).strip(),
                skill=skill,  # type: ignore[arg-type]
                difficulty=difficulty,  # type: ignore[arg-type]
                notes=str(row.get("notes") or "").strip(),
            )
        )
    return items
