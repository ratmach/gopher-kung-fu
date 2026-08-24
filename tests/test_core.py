from app.models import Curriculum, CurriculumItem, Project, ShareGPTExample, ShareGPTTurn
from app.pipeline.distill import planned_count
from app.pipeline.filters import keep_example
from app.store import ProjectStore, slugify, validate_slug
from app.teachers.client import parse_json_payload, strip_think


def test_slugify():
    assert slugify("Gopher Kungfu") == "gopher-kungfu"
    assert validate_slug("sql-specialist") == "sql-specialist"


def test_keep_example_requires_code_for_write():
    bad = ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="Write a mutex"),
            ShareGPTTurn(role="gpt", value="Just use a lock."),
        ],
        meta={"skill": "write"},
    )
    good = ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="Write a mutex"),
            ShareGPTTurn(role="gpt", value="Use:\n```go\nvar mu sync.Mutex\n```\n"),
        ],
        meta={"skill": "write"},
    )
    assert keep_example(bad) is False
    assert keep_example(good) is True


def test_parse_json_and_think():
    raw = "<think>secret</think>\n```json\n{\"items\":[1]}\n```"
    assert strip_think(raw).startswith("```")
    assert parse_json_payload(raw) == {"items": [1]}


def test_parse_json_ignores_go_fence():
    raw = (
        'Here you go\n```go\npackage main\nfunc main() {}\n```\n'
        '{"examples":[{"human":"design a closer","gpt":"```go\\ntype Closer interface { Close() error }\\n```"}]}'
    )
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"


def test_parse_single_quoted_teacher_json():
    raw = "{'examples': [{'human': 'design a closer', 'gpt': 'type Closer interface {}'}]}"
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"
    wrapped = "Sure.\n{'items': [{'id': 'go-1', 'topic': 'Go', 'subtopic': 'ifaces', 'skill': 'write'}]}\n"
    assert parse_json_payload(wrapped)["items"][0]["id"] == "go-1"


def test_parse_unquoted_js_keys():
    raw = '{examples:[{human:"design a closer",gpt:"type Closer interface {}"}]}'
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"
    assert parse_json_payload("{items:[{id:'go-1',topic:'Go',subtopic:'ifaces',skill:'write'}]}")["items"][0]["id"] == "go-1"


def test_distill_assignment_and_planned_count():
    project = Project(slug="gopher-kungfu1", name="Gopher")
    project.distill = {"examples_per_topic": 40, "train_count": 0, "eval_count": 0}
    assert project.distill.examples_per_topic == 40
    assert project.distill.use_batch is False
    raw = Project.model_construct(
        slug="raw",
        name="Raw",
        distill={"examples_per_topic": 12, "train_count": 0, "eval_count": 0},
    )
    curriculum = Curriculum(
        items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")]
    )
    assert planned_count(raw, curriculum) >= 8


def test_read_project_strips_bom(tmp_path):
    project = Project(slug="bom-test", name="BOM")
    path = tmp_path / "project.json"
    path.write_bytes(b"\xef\xbb\xbf" + project.model_dump_json().encode("utf-8"))
    got = ProjectStore()._read_project(path)
    assert got.slug == "bom-test"
