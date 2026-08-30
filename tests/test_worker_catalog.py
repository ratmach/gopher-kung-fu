from pathlib import Path

from app.worker_catalog import (
    MAX_CHARS,
    Export,
    build_neighbor_catalog,
    extract_exports,
    format_catalog,
    load_or_build_index,
    neighbor_import_paths,
)


PARSE_SRC = """package csvparse

type Table struct {
	Headers []string
	Rows    [][]string
}

func Parse(path string) (*Table, error) {
	return nil, nil
}
"""

MERGE_SRC = """package merge

type Table struct {
	Headers []string
}

func Merge(tables []Table) (Table, error) {
	return Table{}, nil
}
"""


def _mod(root: Path) -> None:
    (root / "go.mod").write_text("module example.com/csvmerge\n\ngo 1.22\n", encoding="utf-8")
    parse = root / "internal" / "csvparse" / "parse.go"
    parse.parent.mkdir(parents=True)
    parse.write_text(PARSE_SRC, encoding="utf-8")
    merge = root / "internal" / "merge" / "merge.go"
    merge.parent.mkdir(parents=True)
    merge.write_text(MERGE_SRC, encoding="utf-8")
    other = root / "internal" / "other" / "other.go"
    other.parent.mkdir(parents=True)
    other.write_text(
        "package other\n\nfunc Unrelated() int { return 1 }\n\nfunc AlsoPad() string { return \"\" }\n",
        encoding="utf-8",
    )
    testf = root / "cmd" / "csvmerge" / "main_test.go"
    testf.parent.mkdir(parents=True)
    testf.write_text(
        "package main\n\n"
        "import (\n"
        '\t"encoding/json"\n'
        '\t"flag"\n'
        '\t"example.com/csvmerge/internal/csvparse"\n'
        '\t"example.com/csvmerge/internal/missing"\n'
        ")\n\n"
        "func TestRun(t *testing.T) {\n"
        "\t_ = csvparse.Parse\n"
        "\t_ = json.Marshal\n"
        "\t_ = flag.Args\n"
        "}\n",
        encoding="utf-8",
    )


def test_extract_exports_struct_and_func():
    rows = extract_exports(PARSE_SRC, pkg="internal/csvparse", file="internal/csvparse/parse.go")
    names = {item.name: item for item in rows}
    assert "Table" in names
    assert "Headers []string" in names["Table"].text
    assert names["Parse"].text.startswith("func Parse(path string) (*Table, error)")
    assert "func Parse" in names["Parse"].text
    assert names["Parse"].text.endswith("{") is False or "error)" in names["Parse"].text


def test_neighbor_imports_skip_stdlib_and_missing(tmp_path: Path):
    _mod(tmp_path)
    paths = neighbor_import_paths(tmp_path, ["cmd/csvmerge/main.go"])
    assert "internal/csvparse" in paths
    assert "internal/merge" not in paths
    assert "internal/missing" not in paths
    assert all("encoding" not in p and p != "flag" for p in paths)


def test_exact_dump_has_parse_arity(tmp_path: Path):
    _mod(tmp_path)
    text = build_neighbor_catalog(
        tmp_path,
        ["cmd/csvmerge/main.go"],
        spec="parse a CSV file",
        files="cmd/csvmerge/main.go",
    )
    assert "func Parse(path string) (*Table, error)" in text
    assert "type Table struct" in text
    assert "encoding/json" not in text
    assert "# internal/csvparse" in text


def test_tfidf_retrieves_merge_without_import(tmp_path: Path):
    _mod(tmp_path)
    text = build_neighbor_catalog(
        tmp_path,
        ["cmd/csvmerge/main.go"],
        spec="Align and Merge tables by header name using Merge. De-dupe exact aligned rows.",
        files="cmd/csvmerge/main.go",
    )
    assert "func Parse(path string) (*Table, error)" in text
    assert "# also relevant" in text
    assert "func Merge(tables []Table) (Table, error)" in text
    assert "paraphrase" not in text.lower()


def test_missing_neighbor_dir_does_not_crash(tmp_path: Path):
    _mod(tmp_path)
    text = build_neighbor_catalog(tmp_path, ["cmd/csvmerge/main.go"], spec="parse")
    assert isinstance(text, str)


def test_size_cap_drops_packages_not_mid_signature():
    exact = [
        Export(
            pkg=f"internal/p{i}",
            file=f"internal/p{i}/x.go",
            kind="struct",
            name="Blob",
            text="type Blob struct {\n\t" + ("X int\n\t" * 40) + "}",
        )
        for i in range(12)
    ]
    text = format_catalog(exact, [], budget=800, max_packages=8)
    assert text.count("# internal/") <= 8
    assert len(text) <= 800
    assert "type Blob struct" in text


def test_index_rebuilds_after_mtime(tmp_path: Path):
    _mod(tmp_path)
    first = load_or_build_index(tmp_path)
    index_path = tmp_path / ".gopher" / "catalog_index.json"
    assert index_path.is_file()
    parse = tmp_path / "internal" / "csvparse" / "parse.go"
    parse.write_text(PARSE_SRC + "\nfunc Extra() {}\n", encoding="utf-8")
    second = load_or_build_index(tmp_path)
    names = {row["name"] for row in second["chunks"]}
    assert "Extra" in names
    assert first["mtimes"] != second["mtimes"]


def test_format_catalog_respects_max_chars_constant():
    exact = [
        Export(
            pkg="internal/csvparse",
            file="internal/csvparse/parse.go",
            kind="func",
            name="Parse",
            text="func Parse(path string) (*Table, error)",
        )
    ]
    text = format_catalog(exact, [])
    assert len(text) <= MAX_CHARS
