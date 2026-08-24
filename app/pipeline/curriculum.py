from __future__ import annotations

from app.jobs import JobHub
from app.models import Curriculum, CurriculumItem, Project, utcnow
from app.teachers.client import TeacherClient, TeacherError

SKILLS = ["write", "review", "debug", "refactor", "idiom"]

SYSTEM = """You design a narrow coding curriculum for a specialist small language model.
The model will be fine-tuned only on this niche. Stay specific. Do not broaden into unrelated stacks.
Return JSON only: {"items":[{"id":"kebab-id","topic":"...","subtopic":"...","skill":"write|review|debug|refactor|idiom","difficulty":"easy|medium|hard","notes":"what this item should teach"}]}
Cover each selected topic. Mix the five skills. Prefer concrete library/API/idiom work over essays.
Aim for 8-14 items per selected topic, enough to fill a specialist syllabus without going generic.
"""


def _topic_blob(project: Project) -> str:
    lines = [f"- {topic.label} ({topic.id})" for topic in project.topics]
    return "\n".join(lines) if lines else "- (none)"


async def generate_curriculum(
    project: Project,
    client: TeacherClient,
    hub: JobHub,
    job_id: str,
) -> Curriculum:
    hub.check(job_id)
    await hub.log(job_id, "Asking teacher for a specialist syllabus…", 0.1)
    user = (
        f"Specialist name: {project.name} ({project.slug})\n"
        f"Selected topics:\n{_topic_blob(project)}\n"
        "Keep the niche tight. Every item must help a coding specialist write, review, or debug in these topics."
    )
    raw = await client.chat_json(SYSTEM, user, temperature=0.3)
    items = _coerce_items(raw)
    if not items:
        raise TeacherError("teacher returned an empty curriculum")
    curriculum = Curriculum(items=items, generated_at=utcnow())
    await hub.log(job_id, f"Syllabus ready: {len(items)} items.", 1.0)
    return curriculum


def _coerce_items(raw: object) -> list[CurriculumItem]:
    payload = raw
    if isinstance(raw, dict):
        payload = raw.get("items") or raw.get("curriculum") or []
    if not isinstance(payload, list):
        raise TeacherError("curriculum JSON must contain an items array")
    items: list[CurriculumItem] = []
    seen: set[str] = set()
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
        if ident in seen:
            ident = f"{ident}-{index+1}"
        seen.add(ident)
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
