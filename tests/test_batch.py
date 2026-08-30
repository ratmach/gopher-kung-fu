import asyncio
import json
from pathlib import Path

import httpx

from app.jobs import JobHub
from app.models import Curriculum, CurriculumItem, Job, Project
from app.pipeline.distill import _plan_wave, distill_project
from app.teachers.client import (
    BatchChatRequest,
    BatchChatResult,
    TeacherClient,
    TeacherError,
    batch_api_url,
    strip_batch_suffix,
    teacher_supports_batch,
)


def test_batch_helpers():
    openrouter = Project(
        slug="or",
        name="OR",
        teacher_preset="openrouter",
        teacher_base_url="https://openrouter.ai/api/v1",
        teacher_model="openai/gpt-4o:batch",
    )
    deepseek = Project(slug="ds", name="DS", teacher_preset="deepseek")
    custom = Project(
        slug="cu",
        name="CU",
        teacher_preset="custom",
        teacher_base_url="https://openrouter.ai/api/v1",
        teacher_model="deepseek/deepseek-chat",
    )
    assert teacher_supports_batch(openrouter) is True
    assert teacher_supports_batch(deepseek) is False
    assert teacher_supports_batch(custom) is True
    assert strip_batch_suffix("openai/gpt-4o:batch") == "openai/gpt-4o"
    assert strip_batch_suffix("openai/gpt-4o") == "openai/gpt-4o"
    assert batch_api_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/beta/batches"
    client = TeacherClient(openrouter, "sk-test")
    assert client.supports_batch is True
    assert client.batch_model == "openai/gpt-4o"
    assert client.batch_url.endswith("/api/beta/batches")


def test_plan_wave_chunks_quota():
    project = Project(slug="gopher", name="Gopher")
    item = CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")
    planned = _plan_wave(project, [item], {"go-1": 9}, {}, 0)
    assert [count for _cid, _item, count, _prompt in planned] == [4, 4, 1]
    assert planned[0][0] == "go-1__w0__1"


class _MemStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "synthetic").mkdir(parents=True)

    def train_jsonl(self, slug: str) -> Path:
        return self.root / "synthetic" / "train.jsonl"

    def eval_jsonl(self, slug: str) -> Path:
        return self.root / "synthetic" / "eval.jsonl"

    def inbox_jsonl(self, slug: str) -> Path:
        return self.root / "synthetic" / "inbox.jsonl"

    def project_topic_jsonl(self, slug: str, topic: str) -> Path:
        from app.pipeline.library import topic_slug

        return self.root / "synthetic" / "topics" / f"{topic_slug(topic)}.jsonl"

    def library_topic_jsonl(self, topic: str) -> Path:
        from app.pipeline.library import topic_slug

        return self.root / "library" / topic_slug(topic) / "examples.jsonl"


class _MemHub(JobHub):
    def _persist(self, job: Job) -> None:
        return None


def _code_example(tag: str) -> dict:
    return {
        "human": (
            f"Implement Tag() string in pkg/a.go for {tag}. "
            "constraints: stdlib only, no CGO. files: pkg/a.go"
        ),
        "gpt": (
            f"### pkg/a.go\n```go\npackage pkg\n\n"
            f"func Tag() string {{ return {json.dumps(tag)} }}\n```\n"
        ),
    }


def test_distill_batch_path(tmp_path):
    project = Project(
        slug="gopher-batch",
        name="Gopher",
        teacher_preset="openrouter",
        teacher_model="deepseek/deepseek-chat",
        distill={"examples_per_topic": 8, "train_count": 0, "eval_count": 0, "use_batch": True},
    )
    curriculum = Curriculum(items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")])
    store = _MemStore(tmp_path)
    hub = _MemHub()
    job = Job(id="job1", project_slug=project.slug, kind="distill", status="running")
    hub.jobs[job.id] = job

    class FakeClient:
        supports_batch = True

        async def run_chat_batch(self, requests, **_kwargs):
            out = []
            for req in requests:
                payload = {"examples": [_code_example(f"{req.custom_id}-{n}") for n in range(4)]}
                out.append(BatchChatResult(req.custom_id, json.dumps(payload)))
            return out

    train_n, eval_n = asyncio.run(
        distill_project(project, curriculum, FakeClient(), store, hub, job.id)  # type: ignore[arg-type]
    )
    assert train_n + eval_n >= 4
    assert store.train_jsonl(project.slug).is_file()
    assert any("OpenRouter batch distill" in line for line in job.log)


def test_distill_batch_rejected_for_non_openrouter(tmp_path):
    project = Project(
        slug="gopher-live",
        name="Gopher",
        teacher_preset="deepseek",
        distill={"examples_per_topic": 8, "use_batch": True},
    )
    curriculum = Curriculum(items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")])
    hub = _MemHub()
    job = Job(id="job2", project_slug=project.slug, kind="distill", status="running")
    hub.jobs[job.id] = job
    client = TeacherClient(project, "sk-test")
    try:
        asyncio.run(distill_project(project, curriculum, client, _MemStore(tmp_path), hub, job.id))
        assert False, "expected TeacherError"
    except TeacherError as exc:
        assert "not OpenRouter" in str(exc)


class _FakeResponse:
    def __init__(self, status_code: int, data: dict) -> None:
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> dict:
        return self._data


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        self.posts: list[tuple[str, dict | None]] = []
        self.gets = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url: str, json: dict | None = None, headers=None):
        self.posts.append((url, json))
        if str(url).endswith("/cancel"):
            return _FakeResponse(200, {"status": "cancelling"})
        return _FakeResponse(
            202,
            {"id": "batch_1", "status": "validating", "request_counts": {"total": 1, "completed": 0}},
        )

    async def get(self, url: str, headers=None):
        self.gets += 1
        if self.gets == 1:
            return _FakeResponse(
                200,
                {"id": "batch_1", "status": "in_progress", "request_counts": {"total": 1, "completed": 0}},
            )
        return _FakeResponse(
            200,
            {
                "id": "batch_1",
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1},
                "results": [
                    {
                        "custom_id": "req-1",
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": '{"ok": true}'}}]},
                        },
                        "error": None,
                    }
                ],
            },
        )


def test_run_chat_batch_submit_and_poll(monkeypatch):
    fake = _FakeAsyncClient()

    def factory(*args, **kwargs):
        return fake

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    project = Project(
        slug="or",
        name="OR",
        teacher_preset="openrouter",
        teacher_base_url="https://openrouter.ai/api/v1",
        teacher_model="openai/gpt-4o:batch",
    )
    client = TeacherClient(project, "sk-test")
    results = asyncio.run(
        client.run_chat_batch(
            [BatchChatRequest(custom_id="req-1", system="sys", user="hi")],
            poll_seconds=0.01,
        )
    )
    assert results[0].text and "ok" in results[0].text
    submit = fake.posts[0][1]
    assert list(submit.keys())[:2] == ["endpoint", "model"]
    assert submit["model"] == "openai/gpt-4o"
    assert submit["endpoint"] == "/v1/chat/completions"
    assert fake.gets >= 1
    assert len(fake.posts) == 1
