import subprocess
from pathlib import Path

from app.worker_intel import (
    TypeMiss,
    apply_goimports,
    apply_unused_var_fixes,
    enrich_test_error,
    format_member_hint,
    import_aliases,
    lookup_members,
    members_from_godoc,
    members_from_source,
    parse_type_errors,
    parse_unused_vars,
    patch_unused_line,
    rank_members,
)


GC_NO_METHOD = (
    "internal/hello/hello.go:10:2: c.Foo undefined (type *hello.Client has no field or method Foo)"
)
GC_STDLIB = (
    "./buf.go:6:2: buf.Foo undefined (type *bytes.Buffer has no field or method Foo)"
)
GC_PKG = "store.go:4:2: undefined: json.Marshalx"


def test_parse_type_errors():
    misses = parse_type_errors(GC_NO_METHOD)
    assert len(misses) == 1
    assert misses[0].want == "Foo"
    assert misses[0].type_str == "*hello.Client"
    assert misses[0].kind == "member"

    std = parse_type_errors(GC_STDLIB)
    assert std[0].type_str == "*bytes.Buffer"

    pkg = parse_type_errors(GC_PKG)
    assert pkg[0].kind == "pkg"
    assert pkg[0].expr == "json"
    assert pkg[0].want == "Marshalx"


def test_parse_ignores_plain_undefined_and_unused():
    assert parse_type_errors("x.go:3:2: declared and not used: h") == []
    assert parse_type_errors("x.go:3:2: undefined: NewClient") == []


def test_patch_unused_line_range_index_only():
    assert patch_unused_line('\tbuf := []byte("x")\n', "buf") is None
    assert patch_unused_line("\tbuf, err := r.Read(p)\n", "buf") is None
    assert patch_unused_line("\tfor idx, row := range rows {\n", "idx") == "\tfor _, row := range rows {\n"
    assert patch_unused_line("\tfor idx, row := range rows {\n", "row") is None
    assert patch_unused_line("\tfor idx := range rows {\n", "idx") == "\tfor _ = range rows {\n"
    assert patch_unused_line("\tvar buf []byte\n", "buf") is None


def test_parse_and_apply_unused_vars_range_index_only(tmp_path: Path):
    rel = "internal/csvparse/parse.go"
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)
    dest.write_text(
        "package csvparse\n\nfunc Parse(rows []string) {\n"
        "\tbuf := []byte{1}\n"
        "\tfor idx, row := range rows {\n"
        "\t\t_ = row\n"
        "\t}\n"
        "}\n",
        encoding="utf-8",
    )
    output = (
        "internal/csvparse/parse.go:4:2: declared and not used: buf\n"
        "internal\\csvparse\\parse.go:5:2: declared and not used: idx\n"
    )
    hits = parse_unused_vars(output)
    assert hits[0][2] == "buf"
    assert apply_unused_var_fixes(tmp_path, [rel], output) == [rel]
    text = dest.read_text(encoding="utf-8")
    assert "buf :=" in text
    assert "for _, row := range" in text
    assert "for idx," not in text


def test_apply_unused_vars_skips_accumulator(tmp_path: Path):
    rel = "u.go"
    dest = tmp_path / rel
    dest.write_text("package p\n\nfunc F() {\n\tbuf := []byte{1}\n}\n", encoding="utf-8")
    output = "u.go:4:2: declared and not used: buf"
    assert apply_unused_var_fixes(tmp_path, [rel], output) == []
    assert "buf :=" in dest.read_text(encoding="utf-8")


def test_import_aliases_and_v_suffix():
    source = """
package p

import (
    "bytes"
    redis "github.com/redis/go-redis/v9"
    "encoding/json"
)
"""
    aliases = import_aliases(source)
    assert aliases["bytes"] == "bytes"
    assert aliases["redis"] == "github.com/redis/go-redis/v9"
    assert aliases["json"] == "encoding/json"


def test_members_from_source():
    src = """
package hello

type Client struct {
    Timeout int
    secret  int
}

func (c *Client) Greet() string { return "hi" }
func (c Client) Close() {}
func (c *Client) hidden() {}
func (c *Other) Skip() {}
"""
    names = members_from_source([src], "*Client")
    assert names == ["Greet", "Close", "Timeout"]


def test_members_from_godoc_and_rank():
    doc = """
type Buffer struct{ ... }
func NewBuffer(buf []byte) *Buffer
func (b *Buffer) Bytes() []byte
func (b *Buffer) Write(p []byte) (n int, err error)
"""
    methods = members_from_godoc(doc)
    assert methods == ["Bytes", "Write"]
    iface = members_from_godoc(
        "type Writer interface {\n\tWrite(p []byte) (n int, error)\n}\n"
    )
    assert "Write" in iface
    funcs = members_from_godoc(doc, funcs=True)
    assert funcs[0] == "NewBuffer"
    ranked = rank_members("Foo", ["Bytes", "Write", "Grow"])
    assert ranked[0] in {"Bytes", "Write", "Grow"}
    assert rank_members("Byt", ["Write", "Bytes"])[0] == "Bytes"


def test_format_member_hint_did_you_mean():
    miss = TypeMiss(expr="c", want="Foo", type_str="*Client")
    hint = format_member_hint(miss, ["Greet", "Close"])
    assert "members of *Client" in hint
    assert "Greet" in hint
    assert "Close" in hint


def test_lookup_skips_godoc_when_local_type_has_members(tmp_path: Path, monkeypatch):
    from app import worker_intel as intel

    calls: list[str] = []

    def boom(root, query):
        calls.append(query)
        raise AssertionError("go doc should not run for a local type")

    monkeypatch.setattr(intel, "_run_go_doc", boom)
    src = "package hello\n\ntype Client struct{}\n\nfunc (c *Client) Greet() string { return \"hi\" }\n"
    miss = TypeMiss(expr="c", want="Foo", type_str="*hello.Client")
    names = lookup_members(miss, root=tmp_path, sources=[src], written=["internal/hello/hello.go"])
    assert names == ["Greet"]
    assert calls == []


def test_lookup_uses_godoc_for_imported_type(tmp_path: Path, monkeypatch):
    from app import worker_intel as intel

    monkeypatch.setattr(
        intel,
        "_run_go_doc",
        lambda root, query: "func (b *Buffer) Bytes() []byte\nfunc (b *Buffer) Write(p []byte) (int, error)\n",
    )
    src = 'package p\n\nimport "bytes"\n\nfunc F(b *bytes.Buffer) { b.Foo() }\n'
    miss = TypeMiss(expr="b", want="Foo", type_str="*bytes.Buffer")
    names = lookup_members(miss, root=tmp_path, sources=[src], written=["buf.go"])
    assert "Bytes" in names
    assert "Write" in names


def test_enrich_uses_local_members(tmp_path: Path, monkeypatch):
    from app import worker_intel as intel

    monkeypatch.setattr(intel, "_run_go_doc", lambda root, query: "")
    rel = "internal/hello/hello.go"
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)
    dest.write_text(
        "package hello\n\ntype Client struct{}\n\n"
        "func (c *Client) Greet() string { return \"hi\" }\n",
        encoding="utf-8",
    )
    text = enrich_test_error(GC_NO_METHOD, root=tmp_path, written=[rel])
    assert "Language server" in text
    assert "Greet" in text
    assert GC_NO_METHOD.split("\n")[0] in text


def test_enrich_unused_var_asks_to_use_it(tmp_path: Path):
    rel = "x.go"
    (tmp_path / rel).write_text("package x\n", encoding="utf-8")
    err = "x.go:3:2: declared and not used: h"
    text = enrich_test_error(err, root=tmp_path, written=[rel])
    assert err in text
    assert "use the named local" in text
    assert "never `_ = buf`" in text


def test_apply_goimports_no_tool(tmp_path: Path, monkeypatch):
    from app import worker_intel as intel

    (tmp_path / "a.go").write_text("package a\n", encoding="utf-8")
    monkeypatch.setattr(intel, "_imports_cmd", lambda: None)
    assert apply_goimports(tmp_path, ["a.go"]) == []


def test_apply_goimports_invokes_tool(tmp_path: Path, monkeypatch):
    from app import worker_intel as intel

    (tmp_path / "a.go").write_text("package a\n", encoding="utf-8")
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(intel, "_imports_cmd", lambda: ["goimports", "-w"])
    monkeypatch.setattr(intel.subprocess, "run", fake_run)
    assert apply_goimports(tmp_path, ["a.go"]) == ["a.go"]
    assert seen[0][:2] == ["goimports", "-w"]
    assert seen[0][-1].endswith("a.go")
