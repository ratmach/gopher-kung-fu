import asyncio
import re

from app.jobs import JobHub
from app.models import Job, Project, TopicRef
from app.pipeline.curriculum import (
    TEACHER_BATCH,
    clamp_items_per_topic,
    generate_curriculum,
    syntax_debug_slots,
)


class _MemHub(JobHub):
    def _persist(self, job: Job) -> None:
        return None


class _FakeTeacher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def chat_json(self, system: str, user: str, temperature: float = 0.3) -> dict:
        self.calls.append((system, user))
        match = re.search(r"exactly (\d+)", user)
        need = int(match.group(1)) if match else 1
        wave = len(self.calls)
        return {
            "items": [
                {
                    "id": f"item-{wave}-{i}",
                    "topic": "Go",
                    "subtopic": f"subtopic-{wave}-{i}",
                    "skill": "write",
                    "difficulty": "medium",
                    "notes": "mutex",
                }
                for i in range(need)
            ]
        }


def _project(**extra) -> Project:
    return Project(
        slug="gopher-kungfu",
        name="Gopher",
        topics=[
            TopicRef(id="go", label="Go"),
            TopicRef(id="custom:gin", label="gin", custom=True),
        ],
        **extra,
    )


def test_clamp_items_per_topic():
    assert clamp_items_per_topic(12) == 12
    assert clamp_items_per_topic(1) == 4
    assert clamp_items_per_topic(999) == 80
    assert clamp_items_per_topic("nope") == 12


def test_syntax_debug_slots():
    assert syntax_debug_slots(4) == 1
    assert syntax_debug_slots(12) == 2
    assert syntax_debug_slots(16) == 3


def test_curriculum_system_asks_illegal_go_debug():
    hub = _MemHub()
    job = Job(id="c-debug", project_slug="gopher-kungfu", kind="curriculum", status="running")
    hub.jobs[job.id] = job
    teacher = _FakeTeacher()
    project = _project(items_per_topic=4)
    asyncio.run(generate_curriculum(project, teacher, hub, job.id))  # type: ignore[arg-type]
    systems = [sys for sys, _ in teacher.calls]
    assert systems
    assert any("illegal Go" in sys and "for/else" in sys and "verbatim" in sys for sys in systems)
    assert any("cannot convert 0" in sys for sys in systems)
    assert any("declared and not used" in sys for sys in systems)
    assert any("Write(r)" in sys or "[]byte(string(r))" in sys for sys in systems)
    assert any("map[string]int" in sys for sys in systems)


def test_generate_curriculum_uses_items_per_topic():
    hub = _MemHub()
    job = Job(id="c1", project_slug="gopher-kungfu", kind="curriculum", status="running")
    hub.jobs[job.id] = job
    teacher = _FakeTeacher()
    project = _project(items_per_topic=20)
    result = asyncio.run(generate_curriculum(project, teacher, hub, job.id))  # type: ignore[arg-type]
    assert len(result.items) == 40
    assert teacher.calls
    assert all("exactly" in user for _, user in teacher.calls)
    sizes = [int(re.search(r"exactly (\d+)", user).group(1)) for _, user in teacher.calls]
    assert max(sizes) <= TEACHER_BATCH
    assert sum(sizes) == 34
    assert any(item.id == "go-return-zero-as-error" for item in result.items)
    assert any(item.id == "go-writebyte-eq-zero" for item in result.items)
    assert any(item.id == "go-write-rune-as-bytes" for item in result.items)
    assert any(item.id == "go-map-type-reassign" for item in result.items)
    assert any(item.id == "go-declared-not-used" for item in result.items)
    assert any(item.id == "go-unused-range-index" for item in result.items)
    topics = {item.topic for item in result.items}
    assert "Go" in topics
    assert "gin" in topics
    assert any("20 syllabus items × 2 topics" in line for line in job.log)


def test_catalog_includes_go_compiler():
    from app.catalog import flatten_topics

    ids = {row["id"] for row in flatten_topics()}
    assert "go" in ids
    assert "go-compiler" in ids


def test_curriculum_go_compiler_topic_seeds():
    hub = _MemHub()
    job = Job(id="c-compiler", project_slug="gopher-kungfu", kind="curriculum", status="running")
    hub.jobs[job.id] = job
    teacher = _FakeTeacher()
    project = Project(
        slug="gopher-kungfu",
        name="Gopher",
        topics=[TopicRef(id="go-compiler", label="Go compiler")],
        items_per_topic=8,
    )
    result = asyncio.run(generate_curriculum(project, teacher, hub, job.id))  # type: ignore[arg-type]
    ids = {item.id for item in result.items}
    assert "go-return-zero-as-error" in ids
    assert "go-writebyte-eq-zero" in ids
    assert "go-write-rune-as-bytes" in ids
    assert "go-map-type-reassign" in ids
    assert "go-declared-not-used" in ids
    assert "go-unused-range-index" in ids
    assert "go-err-eq-zero" in ids
    assert all(item.topic == "Go compiler" for item in result.items)
    assert len(result.items) == 8
    sizes = [int(re.search(r"exactly (\d+)", user).group(1)) for _, user in teacher.calls]
    assert sizes == [1]
    assert any("Go compiler diagnostics" in sys for sys, _ in teacher.calls)

