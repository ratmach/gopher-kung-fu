from pathlib import Path

import pytest

from app.worker_fs import ApplyError, apply_files, parse_specialist_files, resolve_under, workspace_root
from app.worker_job import implement_and_apply, packages_for_go_test


SAMPLE = """
### internal/hello/hello.go
```go
package hello

func Greet() string { return "hi" }
```

Done.
"""


def test_parse_specialist_files():
    files = parse_specialist_files(SAMPLE)
    assert len(files) == 1
    assert files[0].rel == "internal/hello/hello.go"
    assert "func Greet()" in files[0].body
    assert files[0].body.endswith("\n")


def test_parse_rejects_escape():
    with pytest.raises(ApplyError, match="escape"):
        parse_specialist_files("### ../secret.go\n```go\npackage x\n```\n")


def test_resolve_stays_in_workspace(tmp_path: Path):
    root = tmp_path.resolve()
    dest = resolve_under(root, "internal/hello/hello.go")
    assert str(dest).startswith(str(root))
    with pytest.raises(ApplyError):
        resolve_under(root, r"C:\Windows\x.go")


def test_apply_writes_and_freeze(tmp_path: Path):
    files = parse_specialist_files(
        "### pkg/a.go\n```go\npackage pkg\n```\n\n### pkg/a_test.go\n```go\npackage pkg\n```\n"
    )
    written, skipped = apply_files(tmp_path, files, freeze_tests=True)
    assert written == ["pkg/a.go"]
    assert skipped == ["pkg/a_test.go"]
    assert (tmp_path / "pkg" / "a.go").read_text(encoding="utf-8") == "package pkg\n"
    assert not (tmp_path / "pkg" / "a_test.go").exists()


def test_workspace_required(monkeypatch):
    monkeypatch.delenv("GOPHER_WORKSPACE", raising=False)
    with pytest.raises(ApplyError, match="workspace is required"):
        workspace_root("")


def test_implement_and_apply_writes_summary(tmp_path: Path):
    def fake_consult(question, *args, **kwargs):
        return SAMPLE

    def fake_test(root, written):
        return True, "ok"

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert "Wrote 1 file" in summary
    assert "internal/hello/hello.go" in summary
    assert "PASS" in summary
    assert "func Greet" not in summary
    assert (tmp_path / "internal" / "hello" / "hello.go").is_file()


def test_implement_retries_passes_error(tmp_path: Path):
    questions: list[str] = []

    def fake_consult(question, *args, **kwargs):
        questions.append(question)
        return SAMPLE

    def fake_test(root, written):
        return len(questions) >= 2, "undefined: Foo"

    summary = implement_and_apply(
        "greeter",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert "undefined: Foo" in questions[1]
    assert "PASS" in summary


def test_apply_false_returns_source(tmp_path: Path):
    def fake_consult(question, *args, **kwargs):
        return SAMPLE

    text = implement_and_apply(
        "greeter",
        apply=False,
        consult_fn=fake_consult,
    )
    assert "func Greet" in text


def test_packages_for_go_test():
    assert packages_for_go_test(["internal/hello/hello.go"]) == ["./internal/hello"]
    assert packages_for_go_test(["main.go"]) == ["."]
