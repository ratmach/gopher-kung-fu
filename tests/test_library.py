from app.models import ShareGPTExample, ShareGPTTurn
from app.pipeline.jsonl import read_jsonl
from app.pipeline.library import (
    append_unique,
    example_fingerprint,
    promote_examples,
    topic_slug,
)


def _row(human: str, gpt: str, topic: str = "Go") -> ShareGPTExample:
    return ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value=human),
            ShareGPTTurn(role="gpt", value=gpt),
        ],
        meta={"topic": topic, "skill": "write"},
    )


def test_topic_slug():
    assert topic_slug("Go") == "go"
    assert topic_slug("custom:gin") == "custom-gin"
    assert topic_slug("  ") == "general"


def test_promote_dedup_and_project_snapshot(tmp_path):
    first = _row("spec a", "```go\npackage a\n```")
    again = _row("spec a", "```go\npackage a\n```")
    other = _row("spec b", "```go\npackage b\n```", topic="SQL")

    lib_root = tmp_path / "library"
    proj_root = tmp_path / "topics"

    def lib_path(topic: str):
        return lib_root / topic_slug(topic) / "examples.jsonl"

    def proj_path(topic: str):
        return proj_root / f"{topic_slug(topic)}.jsonl"

    added = promote_examples(
        [first, again, other],
        library_topic_jsonl=lib_path,
        project_topic_jsonl=proj_path,
    )
    assert added["go"] == 1
    assert added["sql"] == 1
    go_lib = read_jsonl(lib_path("Go"))
    assert len(go_lib) == 1
    assert example_fingerprint(go_lib[0]) == example_fingerprint(first)
    assert len(read_jsonl(proj_path("Go"))) == 1

    extra = _row("spec c", "```go\npackage c\n```")
    assert append_unique(lib_path("Go"), [extra, first]) == 1
    assert len(read_jsonl(lib_path("Go"))) == 2
