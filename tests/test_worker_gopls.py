from dataclasses import dataclass
from pathlib import Path

from app.worker_gopls import missing_method_from_message, type_from_missing_message, unused_name_from_message
from app.worker_intel import apply_gopls_fixes, members_from_godoc, pick_selector_fix


@dataclass
class FakeDiag:
    message: str
    line: int
    character: int
    end_line: int
    end_character: int


class FakeGopls:
    def __init__(self, diags: list[FakeDiag], names: list[str]) -> None:
        self.diags = diags
        self.names = names
        self.synced: list[str] = []
        self.closed = False

    def sync_file(self, rel: str) -> None:
        self.synced.append(rel)

    def diagnostics_for(self, rel: str) -> list[FakeDiag]:
        return list(self.diags)

    def completions_at(self, rel: str, line: int, character: int) -> list[str]:
        return list(self.names)

    def close(self) -> None:
        self.closed = True


def test_missing_method_and_type_from_gc_message():
    msg = "w.WriteRune undefined (type io.Writer has no field or method WriteRune)"
    assert missing_method_from_message(msg) == "WriteRune"
    assert type_from_missing_message(msg) == "io.Writer"
    assert unused_name_from_message("declared and not used: buf") == "buf"
    assert unused_name_from_message("x.go:3:2: declared and not used: idx") == "idx"


def test_pick_selector_fix_writerune_to_write():
    assert pick_selector_fix("WriteRune", ["Write"]) == "Write"
    assert pick_selector_fix("Foo", ["Greet", "Close"]) is None
    assert pick_selector_fix("Write", ["Write"]) is None


def test_members_from_godoc_interface():
    doc = """
type Writer interface {
	Write(p []byte) (n int, error)
}
"""
    assert "Write" in members_from_godoc(doc)


def test_apply_gopls_fixes_rewrites_invented_selector(tmp_path: Path):
    rel = "write.go"
    src = (
        "package p\n\n"
        "import \"io\"\n\n"
        "func F(w io.Writer, r rune) {\n"
        "\tw.WriteRune(r)\n"
        "}\n"
    )
    (tmp_path / rel).write_text(src, encoding="utf-8")
    # Line 5 is `	w.WriteRune(r)` — 0-based line 5? 
    # 0 package, 1 empty, 2 import, 3 empty, 4 func, 5 tab w.WriteRune
    line = 5
    col = src.splitlines()[line].index("WriteRune")
    session = FakeGopls(
        [
            FakeDiag(
                message="w.WriteRune undefined (type io.Writer has no field or method WriteRune)",
                line=line,
                character=col,
                end_line=line,
                end_character=col + len("WriteRune"),
            )
        ],
        ["Write"],
    )
    changed = apply_gopls_fixes(tmp_path, [rel], session)
    assert changed == [rel]
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert "WriteRune" not in text
    assert "w.Write(" in text
    assert session.synced


def test_implement_applies_gopls_before_go_test(tmp_path: Path):
    from app.worker_job import implement_and_apply

    rel = "internal/hello/hello.go"
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)

    body = """### internal/hello/hello.go
```go
package hello

func Use(w interface{ Write([]byte) (int, error) }) {
	w.WriteRune('x')
}
```
"""

    def fake_consult(question, *args, **kwargs):
        return body

    seen_src: list[str] = []

    def fake_test(root, written):
        text = (root / rel).read_text(encoding="utf-8")
        seen_src.append(text)
        if "WriteRune" in text:
            return False, "should have been rewritten"
        return True, "ok"

    line_src = (
        "package hello\n\n"
        "func Use(w interface{ Write([]byte) (int, error) }) {\n"
        "\tw.WriteRune('x')\n"
        "}\n"
    )
    line = 3
    col = line_src.splitlines()[line].index("WriteRune")
    session = FakeGopls(
        [
            FakeDiag(
                message="w.WriteRune undefined (type interface{Write([]byte) (int, error)} has no field or method WriteRune)",
                line=line,
                character=col,
                end_line=line,
                end_character=col + len("WriteRune"),
            )
        ],
        ["Write"],
    )

    summary = implement_and_apply(
        "writer",
        files=rel,
        workspace=str(tmp_path),
        consult_fn=fake_consult,
        go_test_fn=fake_test,
        gopls_factory=lambda root: session,
    )
    assert "PASS" in summary
    assert "WriteRune" not in seen_src[0]
    assert session.closed


def test_apply_gopls_fixes_blanks_unused_range_index(tmp_path: Path):
    rel = "u.go"
    src = (
        "package p\n\nfunc F(rows []int) int {\n"
        "\tn := 0\n"
        "\tfor idx, v := range rows {\n"
        "\t\tn += v\n"
        "\t}\n"
        "\treturn n\n"
        "}\n"
    )
    (tmp_path / rel).write_text(src, encoding="utf-8")
    line = 4
    col = src.splitlines()[line].index("idx")
    session = FakeGopls(
        [
            FakeDiag(
                message="declared and not used: idx",
                line=line,
                character=col,
                end_line=line,
                end_character=col + 3,
            )
        ],
        [],
    )
    assert apply_gopls_fixes(tmp_path, [rel], session) == [rel]
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert "for _, v := range" in text
    assert "for idx," not in text


def test_apply_gopls_fixes_leaves_unused_accumulator(tmp_path: Path):
    rel = "u.go"
    src = "package p\n\nfunc F() {\n\tbuf := 1\n}\n"
    (tmp_path / rel).write_text(src, encoding="utf-8")
    line = 3
    col = src.splitlines()[line].index("buf")
    session = FakeGopls(
        [
            FakeDiag(
                message="declared and not used: buf",
                line=line,
                character=col,
                end_line=line,
                end_character=col + 3,
            )
        ],
        [],
    )
    assert apply_gopls_fixes(tmp_path, [rel], session) == []
    assert "buf :=" in (tmp_path / rel).read_text(encoding="utf-8")
