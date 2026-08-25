import asyncio
import re

from app.jobs import JobHub
from app.models import Job, Project, TopicRef
from app.pipeline.curriculum import (
    TEACHER_BATCH,
    clamp_items_per_topic,
    generate_curriculum,
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
    assert sum(sizes) == 40
    topics = {item.topic for item in result.items}
    assert "Go" in topics
    assert "gin" in topics
    assert any("20 syllabus items × 2 topics" in line for line in job.log)
