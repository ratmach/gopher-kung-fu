from __future__ import annotations

from app.jobs import JobHub
from app.models import Curriculum, CurriculumItem, Project, TopicRef, utcnow
from app.pipeline.seeds import (
    GC_CONVERT_ZERO,
    GC_MAP_REASSIGN,
    go_compiler_seed_items,
    is_go_compiler_topic,
    is_go_language_topic,
)
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


def syntax_debug_slots(count: int) -> int:
    """How many illegal-Go / real-gc debug items to require in a batch of `count`."""
    n = max(1, int(count))
    return max(1, min(n, (n + 5) // 6))


def _system(count: int, topic_label: str, *, compiler_topic: bool = False) -> str:
    debug_n = syntax_debug_slots(count)
    compiler = ""
    if compiler_topic:
        compiler = (
            " This topic is Go compiler diagnostics. Majority skill=debug repair packets. "
            "Every debug notes field must name a verbatim `gc` line."
        )
    return f"""You design a narrow coding curriculum for a specialist small language model.
The model will be fine-tuned only on this niche. Stay specific. Do not broaden into unrelated stacks.
Return JSON only: {{"items":[{{"id":"kebab-id","topic":"{topic_label}","subtopic":"...","skill":"write|review|debug|refactor|idiom","difficulty":"easy|medium|hard","notes":"what this item should teach"}}]}}
Cover only the topic "{topic_label}". Mix skills but make write and debug the majority. Include some review/refactor/idiom. Prefer concrete library/API/testing work over essays.{compiler}
At least {debug_n} item(s) must be skill=debug whose notes are illegal Go that `gc` rejects (Python-isms: for/else, elif, try/except, def, list comprehensions; `return 0` / `err == 0` / `WriteByte(...) == 0` which gc reports as {GC_CONVERT_ZERO}; `w.Write(r)` with r rune — Write wants []byte, gpt uses `[]byte(string(r))` / `io.WriteString`; `seen := map[string]int{{}}` then assigned map[string][]string — gc reports {GC_MAP_REASSIGN}, gpt declares the map with the stored value type; and `declared and not used: x` — use the named local, `_` only for `for _, v := range`, never `_ = buf`). Distill those as broken file + verbatim compiler/`go test` line → patched compiling file. Not review essays.
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
    compiler_topic = is_go_compiler_topic(topic.id, topic.label)
    if is_go_language_topic(topic.id, topic.label):
        for seed in go_compiler_seed_items(topic.label, compiler_topic=compiler_topic):
            if seed.id in seen or len(items) >= n:
                continue
            seen.add(seed.id)
            items.append(seed)
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
        raw = await client.chat_json(
            _system(need, topic.label, compiler_topic=compiler_topic),
            user,
            temperature=0.3,
        )
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
