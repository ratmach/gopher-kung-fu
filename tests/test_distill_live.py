import asyncio
import json
from pathlib import Path

from app.jobs import JobHub
from app.models import Curriculum, CurriculumItem, Job, Project, ShareGPTExample, ShareGPTTurn
from app.pipeline.distill import distill_project
from app.pipeline.inbox import ExampleInbox, example_key
from app.pipeline.jsonl import example_payload
from app.teachers.client import TeacherError


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


def _job(hub: JobHub, slug: str, job_id: str = "job1") -> Job:
    job = Job(id=job_id, project_slug=slug, kind="distill", status="running")
    hub.jobs[job.id] = job
    return job


def test_inbox_roundtrip(tmp_path):
    path = tmp_path / "inbox.jsonl"
    inbox = ExampleInbox(path)

    async def run():
        row = ShareGPTExample(
            conversations=[
                ShareGPTTurn(role="human", value="q"),
                ShareGPTTurn(role="gpt", value="```go\na\n```"),
            ],
            meta={"item_id": "go-1"},
        )
        key = await inbox.put("go-1", row, limit=8)
        assert key == "go-1__0"
        again = ExampleInbox(path)
        assert again.count_for("go-1") == 1
        assert again.examples_for("go-1")[0].meta["key"] == "go-1__0"

    asyncio.run(run())


def test_live_topic_parallel_and_topic_serial(tmp_path):
    project = Project(
        slug="gopher-par",
        name="Gopher",
        teacher_preset="deepseek",
        status="curriculum",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    curriculum = Curriculum(
        items=[
            CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write"),
            CurriculumItem(id="go-2", topic="Go", subtopic="iface", skill="review"),
            CurriculumItem(id="sql-1", topic="SQL", subtopic="join", skill="write"),
        ]
    )
    store = _MemStore(tmp_path)
    hub = _MemHub()
    job = _job(hub, project.slug)
    order: list[tuple[str, str]] = []
    inflight = 0
    max_inflight = 0

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            nonlocal inflight, max_inflight
            topic = "Go" if "Syllabus item: Go" in user else "SQL"
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            order.append(("start", topic))
            await asyncio.sleep(0.05)
            inflight -= 1
            order.append(("end", topic))
            return {"examples": [_code_example(f"{topic}-{i}") for i in range(8)]}

    train_n, eval_n = asyncio.run(
        distill_project(project, curriculum, FakeClient(), store, hub, job.id)  # type: ignore[arg-type]
    )
    assert train_n + eval_n >= 4
    assert max_inflight >= 2
    first_sql = next(i for i, event in enumerate(order) if event == ("start", "SQL"))
    last_go_end = max(i for i, event in enumerate(order) if event == ("end", "Go"))
    assert last_go_end < first_sql
    inbox = ExampleInbox(store.inbox_jsonl(project.slug))
    assert inbox.count_for("go-1") >= 1
    assert inbox.count_for("go-2") >= 1
    assert store.library_topic_jsonl("Go").is_file()
    assert store.project_topic_jsonl(project.slug, "Go").is_file()
    assert store.library_topic_jsonl("SQL").is_file()
    assert any("Saved go-1__" in line for line in job.log)


def test_distill_resumes_inbox_keys(tmp_path):
    project = Project(
        slug="gopher-resume",
        name="Gopher",
        teacher_preset="deepseek",
        status="error",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    item = CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")
    curriculum = Curriculum(items=[item])
    store = _MemStore(tmp_path)
    path = store.inbox_jsonl(project.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for slot in range(5):
            row = ShareGPTExample(
                conversations=[
                    ShareGPTTurn(role="human", value=f"q{slot}"),
                    ShareGPTTurn(role="gpt", value=f"### pkg/a.go\n```go\npackage pkg\n\nfunc V{slot}() int {{ return {slot} }}\n```\n"),
                ],
                meta={"item_id": "go-1", "topic": "Go", "subtopic": "chan", "skill": "write"},
            )
            fh.write(json.dumps({"key": example_key("go-1", slot), **example_payload(row)}) + "\n")

    prompts: list[str] = []
    hub = _MemHub()
    job = _job(hub, project.slug, "job-resume")

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            prompts.append(user)
            return {"examples": [_code_example(f"more-{i}") for i in range(8)]}

    asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
    assert len(prompts) == 1
    assert "Write 3 diverse examples" in prompts[0]
    inbox = ExampleInbox(path)
    assert inbox.count_for("go-1") == 8
    assert any("Resuming 5 saved" in line for line in job.log)


def test_redo_clears_inbox(tmp_path):
    project = Project(
        slug="gopher-redo",
        name="Gopher",
        teacher_preset="deepseek",
        status="ready_to_train",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    curriculum = Curriculum(items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")])
    store = _MemStore(tmp_path)
    path = store.inbox_jsonl(project.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for slot in range(8):
            row = ShareGPTExample(
                conversations=[
                    ShareGPTTurn(role="human", value=f"q{slot}"),
                    ShareGPTTurn(role="gpt", value=f"### pkg/a.go\n```go\npackage pkg\n\nfunc V{slot}() int {{ return {slot} }}\n```\n"),
                ],
                meta={"item_id": "go-1", "topic": "Go", "subtopic": "chan", "skill": "write"},
            )
            fh.write(json.dumps({"key": example_key("go-1", slot), **example_payload(row)}) + "\n")
    hub = _MemHub()
    job = _job(hub, project.slug, "job-redo")
    prompts: list[str] = []

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            prompts.append(user)
            return {"examples": [_code_example(str(i)) for i in range(8)]}

    asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
    inbox = ExampleInbox(path)
    assert prompts
    assert any("fresh distill" in line for line in job.log)
    assert inbox.count_for("go-1") == 8


def test_redo_fills_only_new_curriculum_item(tmp_path):
    project = Project(
        slug="gopher-delta",
        name="Gopher",
        teacher_preset="deepseek",
        status="exported",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    curriculum = Curriculum(
        items=[
            CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write"),
            CurriculumItem(
                id="go-return-zero-as-error",
                topic="Go compiler",
                subtopic="return 0 is not error",
                skill="debug",
            ),
        ]
    )
    store = _MemStore(tmp_path)
    path = store.inbox_jsonl(project.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for slot in range(8):
            row = ShareGPTExample(
                conversations=[
                    ShareGPTTurn(role="human", value=f"q{slot}"),
                    ShareGPTTurn(role="gpt", value=f"### pkg/a.go\n```go\npackage pkg\n\nfunc V{slot}() int {{ return {slot} }}\n```\n"),
                ],
                meta={"item_id": "go-1", "topic": "Go", "subtopic": "chan", "skill": "write"},
            )
            fh.write(json.dumps({"key": example_key("go-1", slot), **example_payload(row)}) + "\n")
    prompts: list[str] = []
    hub = _MemHub()
    job = _job(hub, project.slug, "job-delta")

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            prompts.append(user)
            return {"examples": [_code_example(f"fix-{i}") for i in range(8)]}

    asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
    inbox = ExampleInbox(path)
    assert inbox.count_for("go-1") == 8
    assert inbox.count_for("go-return-zero-as-error") >= 1
    assert prompts
    assert all("Go compiler" in prompt for prompt in prompts)
    assert all("Write " in prompt and "diverse examples" in prompt for prompt in prompts)
    assert any("Asking teacher for" in line for line in job.log)
    assert any("Resuming 8 saved" in line for line in job.log)
    assert not any("fresh distill" in line for line in job.log)


def test_compiler_gold_passes_compile_gate():
    from app.pipeline.compile_gate import compile_gate_example
    from app.pipeline.filters import keep_example
    from app.pipeline.seeds import gold_examples

    rows = gold_examples()
    assert len(rows) == 8
    unused = next(row for row in rows if row.meta.get("item_id") == "go-declared-not-used")
    gpt = unused.conversations[-1].value
    assert "append" in gpt
    assert "_ = " not in gpt
    rng = next(row for row in rows if row.meta.get("item_id") == "go-unused-range-index")
    assert "for _, row := range" in rng.conversations[-1].value
    write_rune = next(row for row in rows if row.meta.get("item_id") == "go-write-rune-as-bytes")
    write_gpt = write_rune.conversations[-1].value
    assert "[]byte(string(r))" in write_gpt
    assert "w.Write(r)" not in write_gpt
    remap = next(row for row in rows if row.meta.get("item_id") == "go-map-type-reassign")
    remap_gpt = remap.conversations[-1].value
    assert "map[string][]string" in remap_gpt
    assert "map[string]int" not in remap_gpt
    flags = next(row for row in rows if row.meta.get("item_id") == "go-flagset-parse-args")
    flag_gpt = flags.conversations[-1].value
    assert "fs.Parse(args)" in flag_gpt
    assert "Parse(args[1:])" not in flag_gpt
    assert '"output"' in flag_gpt and '"o"' in flag_gpt
    assert '"format"' in flag_gpt and '"f"' in flag_gpt
    for example in rows:
        assert keep_example(example) is True
        ok, reason = compile_gate_example(example)
        if reason.startswith("skip: go missing"):
            continue
        assert ok is True, reason



def test_live_records_before_teacher_error(tmp_path):
    project = Project(
        slug="gopher-err",
        name="Gopher",
        teacher_preset="deepseek",
        status="curriculum",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    curriculum = Curriculum(
        items=[
            CurriculumItem(id="go-ok", topic="Go", subtopic="ok", skill="write"),
            CurriculumItem(id="go-bad", topic="Go", subtopic="bad", skill="write"),
        ]
    )
    store = _MemStore(tmp_path)
    hub = _MemHub()
    job = _job(hub, project.slug, "job-err")

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            if "Syllabus item: Go — bad" in user:
                raise TeacherError("boom")
            return {"examples": [_code_example(str(i)) for i in range(8)]}

    try:
        asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
    except TeacherError:
        pass
    inbox = ExampleInbox(store.inbox_jsonl(project.slug))
    assert inbox.count_for("go-ok") >= 4
    assert inbox.count_for("go-bad") == 0


def test_auth_error_stops_distill(tmp_path):
    project = Project(
        slug="gopher-401",
        name="Gopher",
        teacher_preset="deepseek",
        status="curriculum",
        distill={"examples_per_topic": 8, "use_batch": False},
    )
    curriculum = Curriculum(
        items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")]
    )
    store = _MemStore(tmp_path)
    hub = _MemHub()
    job = _job(hub, project.slug, "job-401")

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            raise TeacherError('teacher HTTP 401: {"error":{"message":"User not found.","code":401}}')

    try:
        asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
        assert False, "expected TeacherError"
    except TeacherError as exc:
        assert "401" in str(exc)
    assert any("Teacher error on go-1" in line for line in job.log)


def test_live_asks_remaining_quota(tmp_path):
    project = Project(
        slug="gopher-live-quota",
        name="Gopher",
        teacher_preset="deepseek",
        status="curriculum",
        distill={"examples_per_topic": 9, "use_batch": False},
    )
    item = CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")
    curriculum = Curriculum(items=[item])
    store = _MemStore(tmp_path)
    hub = _MemHub()
    job = _job(hub, project.slug, "job-live-quota")
    prompts: list[str] = []

    class FakeClient:
        timeout = 5.0

        async def chat_json(self, system, user, *, temperature=0.4, http=None):
            prompts.append(user)
            return {"examples": [_code_example(f"c-{i}") for i in range(9)]}

    asyncio.run(distill_project(project, curriculum, FakeClient(), store, hub, job.id))  # type: ignore[arg-type]
    assert len(prompts) == 1
    assert "Write 9 diverse examples" in prompts[0]
    inbox = ExampleInbox(store.inbox_jsonl(project.slug))
    assert inbox.count_for("go-1") == 9
    assert any("Asking teacher for 9 on chan" in line for line in job.log)
