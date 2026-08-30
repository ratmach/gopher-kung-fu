from __future__ import annotations

from app.models import CurriculumItem, ShareGPTExample, ShareGPTTurn

GC_CONVERT_ZERO = (
    "cannot convert 0 (untyped int constant) to type interface{Error() string}"
)
GC_WRITE_RUNE = "cannot use r (variable of type rune) as []byte value in argument to w.Write"
GC_MAP_REASSIGN = (
    "cannot use later (variable of type map[string][]string) as map[string]int value in assignment"
)

RETURN_ZERO_ITEM_ID = "go-return-zero-as-error"
WRITEBYTE_ZERO_ITEM_ID = "go-writebyte-eq-zero"
WRITE_RUNE_ITEM_ID = "go-write-rune-as-bytes"
ERR_EQ_ZERO_ITEM_ID = "go-err-eq-zero"
UNUSED_VAR_ITEM_ID = "go-declared-not-used"
UNUSED_RANGE_ITEM_ID = "go-unused-range-index"
MAP_REASSIGN_ITEM_ID = "go-map-type-reassign"
FLAGSET_ITEM_ID = "go-flagset-parse-args"

COMPILER_SEED_IDS = (
    RETURN_ZERO_ITEM_ID,
    WRITEBYTE_ZERO_ITEM_ID,
    WRITE_RUNE_ITEM_ID,
    UNUSED_VAR_ITEM_ID,
    UNUSED_RANGE_ITEM_ID,
    ERR_EQ_ZERO_ITEM_ID,
    MAP_REASSIGN_ITEM_ID,
)


def is_go_language_topic(topic_id: str = "", topic_label: str = "") -> bool:
    ident = (topic_id or "").strip().lower()
    label = (topic_label or "").strip().lower()
    if ident in {"go", "go-compiler"}:
        return True
    return label in {"go", "go compiler"}


def is_go_compiler_topic(topic_id: str = "", topic_label: str = "") -> bool:
    ident = (topic_id or "").strip().lower()
    label = (topic_label or "").strip().lower()
    return ident == "go-compiler" or label == "go compiler"


def go_compiler_seed_items(topic_label: str, *, compiler_topic: bool = False) -> list[CurriculumItem]:
    """Fixed syllabus rows for gc type errors. Injected so the teacher cannot skip them."""
    specs: list[tuple[str, str, str]] = [
        (
            RETURN_ZERO_ITEM_ID,
            "return 0 is not error",
            "Illegal Go: `return 0` from `func () error` (also `err == 0`). "
            f"gc emits: {GC_CONVERT_ZERO}. Distill as broken ### path.go plus that "
            "verbatim line → patched file with `return nil` / `return err`. Not a lecture.",
        ),
        (
            UNUSED_VAR_ITEM_ID,
            "declared and not used",
            "Illegal Go: unused accumulator (`buf :=`, `idx :=`) that the body was supposed "
            "to fill. gc emits: declared and not used: buf. Distill as broken file + that "
            "verbatim line → patched file that USES buf/idx (append, increment, return buf). "
            "Never `_ = buf`.",
        ),
        (
            UNUSED_RANGE_ITEM_ID,
            "unused range index",
            "Illegal Go: unused range index (`for idx, row := range`). "
            "gc emits: declared and not used: idx. Distill as broken file + that verbatim "
            "line → `for _, row := range`. `_` only for a discarded index, never `_ = buf`.",
        ),
        (
            WRITEBYTE_ZERO_ITEM_ID,
            "WriteByte returns error not int",
            "Illegal Go: `if w.WriteByte(c) == 0`. WriteByte returns error. "
            f"gc emits: {GC_CONVERT_ZERO}. Fix: `if err := w.WriteByte(c); err != nil`. "
            "Human is the broken file plus the gc line; gpt is the compiling file.",
        ),
        (
            WRITE_RUNE_ITEM_ID,
            "Write does not take a rune",
            "Illegal Go: `w.Write(r)` with r rune. io.Writer.Write wants []byte. "
            f"gc emits: {GC_WRITE_RUNE}. Distill as broken file + that verbatim line → "
            "`w.Write([]byte(string(r)))` or `io.WriteString(w, string(r))`. "
            "io.Writer has no WriteRune. Not a lecture.",
        ),
        (
            MAP_REASSIGN_ITEM_ID,
            "map value type is fixed at declaration",
            "Illegal Go: `seen := map[string]int{}` then `seen = later` where later is "
            f"map[string][]string. gc emits: {GC_MAP_REASSIGN}. Distill as broken file + "
            "that verbatim line → declare `seen := map[string][]string{}` (the type you store). "
            "Do not keep an int map and assign a string-slice map. Not a lecture.",
        ),
    ]
    if compiler_topic:
        specs.append(
            (
                ERR_EQ_ZERO_ITEM_ID,
                "error compared to 0",
                "Illegal Go: `if err == 0` or `if err != 0` after a call that returns error. "
                f"gc emits: {GC_CONVERT_ZERO}. Fix: `if err != nil`. Repair packet, not an essay.",
            )
        )
    return [
        CurriculumItem(
            id=item_id,
            topic=topic_label,
            subtopic=subtopic,
            skill="debug",
            difficulty="easy",
            notes=notes,
        )
        for item_id, subtopic, notes in specs
    ]


def _example(
    *,
    item_id: str,
    subtopic: str,
    human: str,
    gpt: str,
    topic: str = "Go compiler",
    skill: str = "debug",
    seed: str = "go-compiler",
) -> ShareGPTExample:
    return ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value=human.strip() + "\n"),
            ShareGPTTurn(role="gpt", value=gpt.strip() + "\n"),
        ],
        meta={
            "topic": topic,
            "subtopic": subtopic,
            "skill": skill,
            "difficulty": "easy",
            "item_id": item_id,
            "seed": seed,
        },
    )


def compiler_gold_examples() -> list[ShareGPTExample]:
    """Hand-written repair packets. gpt side must compile; human is the broken unit + gc line."""
    return [
        _example(
            item_id=RETURN_ZERO_ITEM_ID,
            subtopic="return 0 is not error",
            human=f"""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

func Parse(s string) error {{
	if s == "" {{
		return 0
	}}
	return 0
}}
```

parse.go:5:3: {GC_CONVERT_ZERO}
parse.go:7:2: {GC_CONVERT_ZERO}
""",
            gpt="""### parse.go
```go
package csvparse

func Parse(s string) error {
	if s == "" {
		return nil
	}
	return nil
}
```
""",
        ),
        _example(
            item_id=WRITEBYTE_ZERO_ITEM_ID,
            subtopic="WriteByte returns error not int",
            human=f"""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

import "io"

func WriteByte(w io.ByteWriter, c byte) error {{
	if w.WriteByte(c) == 0 {{
		return nil
	}}
	return 0
}}
```

parse.go:7:22: {GC_CONVERT_ZERO}
parse.go:10:9: {GC_CONVERT_ZERO}
""",
            gpt="""### parse.go
```go
package csvparse

import "io"

func WriteByte(w io.ByteWriter, c byte) error {
	if err := w.WriteByte(c); err != nil {
		return err
	}
            return nil
}
```
""",
        ),
        _example(
            item_id=WRITE_RUNE_ITEM_ID,
            subtopic="Write does not take a rune",
            human=f"""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

import "io"

func Put(w io.Writer, r rune) error {{
	_, err := w.Write(r)
	return err
}}
```

parse.go:6:20: {GC_WRITE_RUNE}
""",
            gpt="""### parse.go
```go
package csvparse

import "io"

func Put(w io.Writer, r rune) error {
	_, err := w.Write([]byte(string(r)))
	return err
}
```
""",
        ),
        _example(
            item_id=UNUSED_VAR_ITEM_ID,
            subtopic="declared and not used",
            human="""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

func Parse(s string) []string {
	buf := []string{}
	idx := 0
	return []string{s}
}
```

parse.go:4:2: declared and not used: buf
parse.go:5:2: declared and not used: idx
""",
            gpt="""### parse.go
```go
package csvparse

func Parse(s string) []string {
	buf := []string{}
	idx := 0
	for idx < len(s) {
		buf = append(buf, s[idx:idx+1])
		idx++
	}
	return buf
}
```
""",
        ),
        _example(
            item_id=UNUSED_RANGE_ITEM_ID,
            subtopic="unused range index",
            human="""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

func Join(rows []string) string {
	out := ""
	for idx, row := range rows {
		out += row
	}
	return out
}
```

parse.go:5:2: declared and not used: idx
""",
            gpt="""### parse.go
```go
package csvparse

func Join(rows []string) string {
	out := ""
	for _, row := range rows {
		out += row
	}
	return out
}
```
""",
        ),
        _example(
            item_id=ERR_EQ_ZERO_ITEM_ID,
            subtopic="error compared to 0",
            human=f"""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### parse.go
```go
package csvparse

import "io"

func Flush(w io.Writer) error {{
	_, err := w.Write(nil)
	if err == 0 {{
		return 0
	}}
	return err
}}
```

parse.go:8:5: {GC_CONVERT_ZERO}
parse.go:9:10: {GC_CONVERT_ZERO}
""",
            gpt="""### parse.go
```go
package csvparse

import "io"

func Flush(w io.Writer) error {
	_, err := w.Write(nil)
	if err != nil {
		return err
	}
	return nil
}
```
""",
        ),
        _example(
            item_id=MAP_REASSIGN_ITEM_ID,
            subtopic="map value type is fixed at declaration",
            human=f"""REPAIR. Previous file failed `go test`. Patch the implementation. Frozen tests stay. stdlib only.

### merge.go
```go
package merge

func Union(names []string) map[string][]string {{
	seen := map[string]int{{}}
	out := map[string][]string{{}}
	for _, name := range names {{
		if _, ok := seen[name]; ok {{
			continue
		}}
		later := map[string][]string{{name: {{name}}}}
		seen = later
		out[name] = []string{{name}}
	}}
	return out
}}
```

merge.go:11:10: {GC_MAP_REASSIGN}
""",
            gpt="""### merge.go
```go
package merge

func Union(names []string) map[string][]string {
	seen := map[string][]string{}
	out := map[string][]string{}
	for _, name := range names {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = []string{name}
		out[name] = []string{name}
	}
	return out
}
```
""",
        ),
    ]


def write_gold_examples() -> list[ShareGPTExample]:
    """Hand-written implement packets. gpt side must compile."""
    return [
        _example(
            item_id=FLAGSET_ITEM_ID,
            topic="Go",
            subtopic="FlagSet Parse uses argv as-is",
            skill="write",
            seed="go-write",
            human="""IMPLEMENT. stdlib only. One file.

Spec: run(args []string, stdout io.Writer) error. args is already the command argv (like os.Args[1:]). Register --output/-o and --format/-f. Parse args as-is — do not call Parse(args[1:]). Print output and format to stdout, space-separated.

Constraints: flag.FlagSet, no CGO. Honor the signature.

### cmd/csvmerge/main.go
func run(args []string, stdout io.Writer) error
""",
            gpt="""### cmd/csvmerge/main.go
```go
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
)

func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("csvmerge", flag.ContinueOnError)
	var output string
	var format string
	fs.StringVar(&output, "output", "", "output path")
	fs.StringVar(&output, "o", "", "output path")
	fs.StringVar(&format, "format", "csv", "output format")
	fs.StringVar(&format, "f", "csv", "output format")
	if err := fs.Parse(args); err != nil {
		return err
	}
	_, err := fmt.Fprintf(stdout, "%s %s\\n", output, format)
	return err
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		os.Exit(1)
	}
}
```
""",
        ),
    ]


def gold_examples() -> list[ShareGPTExample]:
    return compiler_gold_examples() + write_gold_examples()
