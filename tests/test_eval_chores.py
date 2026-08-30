import shutil

import pytest

from app.pipeline.eval_chores import list_chores, run_suite


CLAMP = """### internal/clamp/clamp.go
```go
package clamp

func Clamp(n, lo, hi int) int {
	if n < lo {
		return lo
	}
	if n > hi {
		return hi
	}
	return n
}
```
"""

GREET = """### internal/greet/greet.go
```go
package greet

func Greet(name string) string {
	if name == "" {
		return "hello"
	}
	return "hello, " + name
}
```
"""


def test_list_chores():
    ids = {c["id"] for c in list_chores()}
    assert ids >= {"clamp", "greet"}


def test_eval_chores_pass_with_gold():
    if not shutil.which("go"):
        pytest.skip("go binary not on PATH")

    def consult(question, *args, **kwargs):
        if "Clamp" in question:
            return CLAMP
        return GREET

    report = run_suite(consult_fn=consult, retries=3)
    assert report["chores"] == 2
    assert report["pass_at_1"] == 1.0
    assert report["pass_at_3"] == 1.0
