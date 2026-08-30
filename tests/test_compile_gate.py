from app.models import ShareGPTExample, ShareGPTTurn
from app.pipeline.compile_gate import compile_gate_example, extract_go_files
from app.worker_imports import go_imports, import_violations, is_stdlib


def _ex(gpt: str, skill: str = "write") -> ShareGPTExample:
    return ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="spec"),
            ShareGPTTurn(role="gpt", value=gpt),
        ],
        meta={"skill": skill},
    )


def test_extract_go_files():
    gpt = "### pkg/a.go\n```go\npackage pkg\n\nfunc A() {}\n```\n"
    files = extract_go_files(_ex(gpt))
    assert files[0][0] == "pkg/a.go"
    assert "func A()" in files[0][1]


def test_compile_gate_skips_review():
    ok, reason = compile_gate_example(_ex("no fences", skill="review"), go_bin=False)
    assert ok is True
    assert "not implement" in reason


def test_compile_gate_drops_missing_fences():
    ok, reason = compile_gate_example(_ex("```go\nvar x int\n```\n"), go_bin=False)
    assert ok is False
    assert "fences" in reason


def test_compile_gate_drops_third_party():
    gpt = (
        "### pkg/a.go\n```go\npackage pkg\n\n"
        "import \"github.com/gin-gonic/gin\"\nfunc A() {}\n```\n"
    )
    ok, reason = compile_gate_example(_ex(gpt), go_bin=False)
    assert ok is False
    assert "non-stdlib" in reason


def test_compile_gate_keeps_when_go_missing():
    gpt = "### pkg/a.go\n```go\npackage pkg\n\nfunc A() {}\n```\n"
    ok, reason = compile_gate_example(_ex(gpt), go_bin=False)
    assert ok is True
    assert "go missing" in reason


def test_compile_gate_compiles_stdlib():
    gpt = "### pkg/a.go\n```go\npackage pkg\n\nfunc A() string { return \"ok\" }\n```\n"
    ok, reason = compile_gate_example(_ex(gpt))
    if reason.startswith("skip: go missing"):
        return
    assert ok is True, reason


def test_import_violations_stdlib_ok():
    src = 'package p\n\nimport (\n\t"fmt"\n\t"net/http"\n)\n'
    assert go_imports(src) == ["fmt", "net/http"]
    assert is_stdlib("fmt")
    assert not is_stdlib("github.com/gin-gonic/gin")
    assert import_violations(src, allowed_imports=[], forbidden_imports=["C"]) == []


def test_import_violations_forbidden_stdlib():
    src = 'package p\n\nimport "encoding/csv"\n'
    assert import_violations(src, allowed_imports=[], forbidden_imports=["C"]) == []
    hits = import_violations(
        src,
        allowed_imports=[],
        forbidden_imports=["C", "encoding/csv"],
    )
    assert hits == ["encoding/csv"]


def test_forbidden_from_constraints():
    from app.worker_imports import forbidden_from_constraints, resolve_allowlists

    assert "encoding/csv" in forbidden_from_constraints("never import encoding/csv")
    assert forbidden_from_constraints("stdlib only, modernc.org/sqlite allowed") == []
    allowed, forbidden = resolve_allowlists(
        "",
        allowed_imports=[],
        forbidden_imports=["C"],
        constraints="RFC 4180; never import encoding/csv",
    )
    assert allowed == []
    assert "encoding/csv" in forbidden
    assert "C" in forbidden


def test_import_violations_third_party_and_cgo():
    src = 'package p\n\nimport (\n\t"C"\n\t"github.com/gin-gonic/gin"\n)\n'
    hits = import_violations(src, allowed_imports=[], forbidden_imports=["C"])
    assert "C" in hits
    assert "github.com/gin-gonic/gin" in hits
    allowed = import_violations(
        'package p\nimport "modernc.org/sqlite"\n',
        allowed_imports=["modernc.org/sqlite"],
        forbidden_imports=["C"],
    )
    assert allowed == []
