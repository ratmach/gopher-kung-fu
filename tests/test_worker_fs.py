from pathlib import Path

import pytest

from app.worker_fs import (
    ApplyError,
    apply_files,
    history_hash_path,
    map_parsed_to_wanted,
    parse_specialist_files,
    parse_wanted_files,
    previous_attempt_blob,
    record_write_hashes,
    resolve_under,
    workspace_root,
)
from app.worker_job import (
    error_only_in_frozen_tests,
    implement_and_apply,
    packages_for_go_test,
)


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


def test_parse_skips_error_fence_before_go_fence():
    text = """
### internal/merge/merge.go

The compiler said:
```
declared and not used: h
```

```go
package merge

func Merge() {}
```
"""
    files = parse_specialist_files(text)
    assert len(files) == 1
    assert files[0].rel == "internal/merge/merge.go"
    assert "package merge" in files[0].body
    assert "declared and not used" not in files[0].body


def test_parse_does_not_grab_next_file_fence():
    text = """
### a.go
```go
package a
```

### b.go
```go
package b
```
"""
    files = parse_specialist_files(text)
    assert [item.rel for item in files] == ["a.go", "b.go"]
    assert files[0].body == "package a\n"
    assert files[1].body == "package b\n"


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


def test_map_parsed_remaps_basename_and_drops_extras():
    parsed = parse_specialist_files(
        "### merge.go\n```go\npackage merge\n```\n\n### extra.go\n```go\npackage extra\n```\n"
    )
    wanted = parse_wanted_files("internal/merge/merge.go")
    mapped, extras = map_parsed_to_wanted(parsed, wanted)
    assert [item.rel for item in mapped] == ["internal/merge/merge.go"]
    assert mapped[0].body == "package merge\n"
    assert extras == ["extra.go"]


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


def test_implement_passes_neighbor_apis(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module example.com/csvmerge\n\ngo 1.22\n", encoding="utf-8")
    parse = tmp_path / "internal" / "csvparse" / "parse.go"
    parse.parent.mkdir(parents=True)
    parse.write_text(
        "package csvparse\n\ntype Table struct {\n\tHeaders []string\n}\n\n"
        "func Parse(path string) (*Table, error) { return nil, nil }\n",
        encoding="utf-8",
    )
    testf = tmp_path / "cmd" / "csvmerge" / "main_test.go"
    testf.parent.mkdir(parents=True)
    testf.write_text(
        "package main\n\nimport \"example.com/csvmerge/internal/csvparse\"\n\n"
        "func TestRun(t *testing.T) { _ = csvparse.Parse }\n",
        encoding="utf-8",
    )
    seen: dict = {}

    def fake_consult(question, *args, **kwargs):
        seen.update(kwargs)
        return (
            "### cmd/csvmerge/main.go\n```go\npackage main\n\nfunc main() {}\n```\n"
        )

    implement_and_apply(
        "wire the CLI using Parse",
        files="cmd/csvmerge/main.go",
        workspace=str(tmp_path),
        consult_fn=fake_consult,
        go_test_fn=lambda root, written: (True, "ok"),
    )
    apis = seen.get("neighbor_apis") or ""
    assert "func Parse(path string) (*Table, error)" in apis
    assert "type Table struct" in apis


def test_implement_retries_sends_member_list_on_undefined_method(tmp_path: Path):
    kwargs_seen: list[dict] = []
    body = """### internal/hello/hello.go
```go
package hello

type Client struct{}

func (c *Client) Greet() string { return "hi" }

func Use(c *Client) string { return c.Foo() }
```
"""

    def fake_consult(question, *args, **kwargs):
        kwargs_seen.append(kwargs)
        return body

    def fake_test(root, written):
        if len(kwargs_seen) < 2:
            return False, (
                "internal/hello/hello.go:8:2: c.Foo undefined "
                "(type *hello.Client has no field or method Foo)"
            )
        return True, "ok"

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert "Language server" in kwargs_seen[1]["test_error"]
    assert "Greet" in kwargs_seen[1]["test_error"]
    assert "members of" in kwargs_seen[1]["test_error"]
    assert "PASS" in summary


def test_implement_retries_sends_previous_file_and_error(tmp_path: Path):
    questions: list[str] = []
    kwargs_seen: list[dict] = []

    def fake_consult(question, *args, **kwargs):
        questions.append(question)
        kwargs_seen.append(kwargs)
        return SAMPLE

    def fake_test(root, written):
        return len(questions) >= 2, "internal/hello/hello.go:3:2: declared and not used: h"

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert questions[0] == "greeter"
    assert not kwargs_seen[0].get("test_error")
    assert not kwargs_seen[0].get("previous_files")
    assert "declared and not used: h" in kwargs_seen[1]["test_error"]
    assert "func Greet()" in kwargs_seen[1]["previous_files"]
    assert "### internal/hello/hello.go" in kwargs_seen[1]["previous_files"]
    assert "Previous compile/test error" not in questions[1]
    assert "PASS" in summary
    assert "repair used previous file" not in summary


def test_implement_retries_when_frozen_test_does_not_compile(tmp_path: Path):
    calls = {"n": 0}
    kwargs_seen: list[dict] = []

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        kwargs_seen.append(kwargs)
        return SAMPLE

    def fake_test(root, written):
        return False, (
            "internal/hello/hello_test.go:8:2: undefined: Greet\n"
            "FAIL\texample/internal/hello [build failed]"
        )

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        freeze_tests=True,
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert calls["n"] == 3
    assert "Do not edit tests" in kwargs_seen[1]["test_error"]
    assert "undefined: Greet" in kwargs_seen[1]["test_error"]
    assert "repair used previous file" in summary
    assert "FAIL" in summary
    assert "cannot edit those" not in summary


def test_implement_retries_when_assertions_cite_test_file(tmp_path: Path):
    calls = {"n": 0}

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        return SAMPLE

    def fake_test(root, written):
        if calls["n"] < 2:
            return False, (
                "--- FAIL: TestGreet (0.00s)\n"
                "    hello_test.go:12: Greet() = \"nope\", want \"hi\"\n"
                "FAIL"
            )
        return True, "ok"

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        freeze_tests=True,
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert calls["n"] == 2
    assert "PASS" in summary
    assert "frozen *_test.go" not in summary


def test_implement_refuses_paths_outside_files_list(tmp_path: Path):
    def fake_consult(question, *args, **kwargs):
        return "### other.go\n```go\npackage other\n```\n"

    with pytest.raises(ApplyError, match="no requested files"):
        implement_and_apply(
            "greeter",
            files="internal/hello/hello.go",
            workspace=str(tmp_path),
            consult_fn=fake_consult,
            go_test_fn=lambda root, written: (True, "ok"),
        )
    assert not (tmp_path / "other.go").exists()


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


def test_error_only_in_frozen_tests():
    assert error_only_in_frozen_tests(
        "internal/merge/merge_test.go:4:2: declared and not used: h\nFAIL\tmod [build failed]",
        freeze_tests=True,
    )
    assert not error_only_in_frozen_tests(
        "internal/merge/merge.go:12:2: declared and not used: h",
        freeze_tests=True,
    )
    assert not error_only_in_frozen_tests(
        "internal/merge/merge_test.go:4:2: declared and not used: h",
        freeze_tests=False,
    )
    # Assertion failures cite the test file; the impl is still the thing to patch.
    assert not error_only_in_frozen_tests(
        "--- FAIL: TestParse (0.00s)\n    parse_test.go:23: got quoted fields, want Split\nFAIL",
        freeze_tests=True,
    )


def test_frozen_job_rejects_test_paths():
    with pytest.raises(ApplyError, match="cannot request test files"):
        implement_and_apply(
            "greeter",
            files="internal/hello/hello_test.go",
            workspace=".",
            apply=False,
            consult_fn=lambda *a, **k: SAMPLE,
        )


def test_implement_constraint_forbids_encoding_csv(tmp_path: Path):
    calls = {"n": 0}

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        return (
            "### internal/csvparse/parse.go\n```go\n"
            "package csvparse\n\nimport \"encoding/csv\"\n"
            "func Parse(p string) error { return csv.ErrTrailingComma }\n```\n"
        )

    summary = implement_and_apply(
        "parse CSV",
        constraints="stdlib only; never import encoding/csv",
        files="internal/csvparse/parse.go",
        workspace=str(tmp_path),
        allowed_imports=[],
        forbidden_imports=["C"],
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=lambda root, written: (True, "ok"),
    )
    assert calls["n"] == 1
    assert "encoding/csv" in summary
    assert "allowlist" in summary
    assert "PASS" not in summary


def test_implement_allowlist_fails_without_retry(tmp_path: Path):
    calls = {"n": 0}

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        return (
            "### internal/hello/hello.go\n```go\n"
            "package hello\n\nimport \"github.com/gin-gonic/gin\"\n"
            "func Greet() string { return gin.Mode() }\n```\n"
        )

    summary = implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        allowed_imports=[],
        forbidden_imports=["C"],
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=lambda root, written: (True, "ok"),
    )
    assert calls["n"] == 1
    assert "allowlist" in summary
    assert "github.com/gin-gonic/gin" in summary
    assert "PASS" not in summary


def test_run_job_requires_one_impl(tmp_path: Path):
    from app.worker_job import run_job

    def fake_consult(question, *args, **kwargs):
        return SAMPLE

    with pytest.raises(ApplyError, match="exactly one impl"):
        run_job(
            "greeter",
            files="a.go\nb.go",
            workspace=str(tmp_path),
            consult_fn=fake_consult,
        )


def test_previous_attempt_blob(tmp_path: Path):
    path = tmp_path / "merge.go"
    path.write_text("package merge\n", encoding="utf-8")
    blob = previous_attempt_blob(tmp_path, ["merge.go"])
    assert blob.startswith("### merge.go\n```go\npackage merge")


def test_record_write_hashes_appends_only_on_change(tmp_path: Path):
    rel = "internal/csvparse/parse.go"
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)
    dest.write_text("package csvparse\n", encoding="utf-8")
    assert record_write_hashes(tmp_path, [rel], attempt=1) == [rel]
    hist = history_hash_path(tmp_path, rel)
    assert hist.name == "parse.go_history.hash"
    first = hist.read_text(encoding="utf-8").strip().splitlines()
    assert len(first) == 1
    assert first[0].startswith("1 ")
    assert record_write_hashes(tmp_path, [rel], attempt=2) == []
    dest.write_text("package csvparse\n\nfunc Parse() {}\n", encoding="utf-8")
    assert record_write_hashes(tmp_path, [rel], attempt=3) == [rel]
    lines = hist.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("3 ")
    assert lines[0].split()[1] != lines[1].split()[1]


def test_implement_hash_history_one_line_when_repair_echoes(tmp_path: Path):
    calls = {"n": 0}

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        return SAMPLE

    def fake_test(root, written):
        return calls["n"] >= 3, "unused"

    implement_and_apply(
        "greeter",
        files="internal/hello/hello.go",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    hist = history_hash_path(tmp_path, "internal/hello/hello.go")
    lines = hist.read_text(encoding="utf-8").strip().splitlines()
    assert calls["n"] == 3
    assert len(lines) == 1
    assert lines[0].startswith("1 ")


def test_implement_unused_accumulator_goes_to_repair(tmp_path: Path):
    body = """### pkg/a.go
```go
package pkg

func F() {
	buf := []byte("x")
}
```
"""
    calls = {"n": 0}
    kwargs_seen: list[dict] = []

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        kwargs_seen.append(kwargs)
        return body

    def fake_test(root, written):
        return calls["n"] >= 2, "pkg/a.go:4:2: declared and not used: buf"

    summary = implement_and_apply(
        "unused",
        files="pkg/a.go",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert calls["n"] == 2
    assert "PASS" in summary
    assert "declared and not used: buf" in kwargs_seen[1]["test_error"]
    assert "use the named local" in kwargs_seen[1]["test_error"]
    assert "buf :=" in (tmp_path / "pkg/a.go").read_text(encoding="utf-8")


def test_implement_host_fixes_unused_range_index_without_retry(tmp_path: Path):
    body = """### pkg/a.go
```go
package pkg

func F(rows []string) string {
	out := ""
	for idx, row := range rows {
		out += row
	}
	return out
}
```
"""
    calls = {"n": 0}

    def fake_consult(question, *args, **kwargs):
        calls["n"] += 1
        return body

    def fake_test(root, written):
        text = (root / "pkg/a.go").read_text(encoding="utf-8")
        if "for idx," in text:
            return False, "pkg/a.go:5:2: declared and not used: idx"
        return True, "ok"

    summary = implement_and_apply(
        "unused-range",
        files="pkg/a.go",
        workspace=str(tmp_path),
        retries=3,
        consult_fn=fake_consult,
        go_test_fn=fake_test,
    )
    assert calls["n"] == 1
    assert "PASS" in summary
    assert "for _, row := range" in (tmp_path / "pkg/a.go").read_text(encoding="utf-8")
