# Local workers

Kimi plans and locks tests; the **MCP host** writes files, runs `go test`, and retries the SLM. Kimi only sees a short summary.

**Built now:** `run_job` (preferred) and `implement_go` parse `### path.go` fences, write under `workspace` / `GOPHER_WORKSPACE`, freeze `*_test.go` by default, check cartridge import allowlists, run `go test`, retry up to 3 times with the failing file + error, return a summary (not source). Implement farm calls are greedy (`temperature: 0`) at `max_tokens` **4096**. The SLM does not emit tool JSON. Thinking mode stays off.

**Still later:** supervisor fan-out of many jobs in one MCP call, tests generated from plan JSON.

## Why this shape

A 1.7B Q4 cartridge cannot land a compiling package (wrong imports, invented types). Kimi K2.7 Coder already does a feature for about **$1.47**; offloading whole services does not beat that.

What can work:

- **3B–7B-class** student (e.g. Qwen2.5-Coder 3B or Qwen3 4B), not 1.7B, for easy–medium **day-to-day** Go
- **One cartridge = one frozen import set** (`gopher-go-redis`, `gopher-go-kafka`, …)
- **One job = one file or one function**, signatures in `existing_code`
- **`go test` is the judge**, not Kimi’s opinion of the source
- Kimi output = **spec / tests / summary**, never the worker’s implementation

LeetCode-hard is out of scope. Gin+SQL+JWT in one model is out of scope. Do not add MiniCPM5 / Qwen3.5 / Gemma 4 as students for this job.

## Token rule

On Kimi K2.7 Code (OpenRouter ballpark **~$0.67/M input**, **~$3.40/M output**):

| Step | Kimi | Cost |
|---|---|---|
| Short spec (“endpoint abcd, params a,b, JSON, handle errors”) | **output** | cheap |
| Specialist Go in the tool result | **input** | ~5× cheaper than output |
| Kimi `write`s that Go to disk | **output again** | wipes the win |

So: **the MCP host writes**. Do not add MCP `write(path, contents)` for Kimi to fill in — that is the same bill as Kilo’s write.

The SLM does **not** need tool-calling. Teaching Gopher `write_file` JSON is how a small model parrots `update_todo_list`. Tools live in the **host**.

## Layers

```
Kimi (orchestrator)
  plan small jobs
  write/validate tests
  call run_job once per job
  on FAIL after budget: implement yourself
        │
        ▼
Supervisor (run_job in MCP)
  one impl path, freeze tests, cartridge allowlist
  repair budget
        │
        ▼
MCP host
  parse ### path / ```go fences
  write impl (not *_test.go)
  go test
  feed compiler/test errors back to Gopher (2–3×)
  return summary only
        │
        ▼
Farm  POST /v1/chat/completions
  Gopher GGUF: greedy, 4096 new tokens, text in → Go out
```

## Cartridge = frozen stack

`card.json` fields `allowed_imports` (featured packages plus extra modules; stdlib is always allowed) and `forbidden_imports` (always includes `C`). Shipped `gopher-go` lists `encoding/csv` and `encoding/json`. Job `constraints` that say **never import encoding/csv** are still merged into that forbid list. Off-list third-party imports **fail the job** without further specialist retries — including the `implement_and_apply` / `implement_go` path, not only `run_job`.

```text
allowed_imports:  [github.com/redis/go-redis/v9]
forbidden_imports: [C, github.com/go-redis/redis]
```

Kimi’s plan may only name those imports. If the task needs another library, **do not offload**.

Setup probes (no per-OS file-writer zoo):

- `go version` on PATH
- `gopls` on PATH (optional; host uses it to rewrite invented selectors). `GOPHER_GOPLS=0` disables it
- workspace root
- After each write (post goimports/gopls), if the impl bytes changed, append `{attempt} {sha256}` to `.gopher/history/<rel>_history.hash`. Same bytes as last line → no append (repair echoed). Delete `.gopher/history` when you reset an eval.
- farm URL + cartridge id
- `llama-server` flavor is a **farm** concern (CUDA vs CPU vs Metal), not MCP I/O

File I/O is Python/`os` (or Go `os`) once. Do not wrap MCP in `cmd /c` or bash on Windows (stdio already broke that way).

## Job

Small enough for a 3B with compile-retry:

- `spec` — behavior and tests that must pass
- `constraints` — allowlist, no-CGO, paths, names
- `files` — **one** impl path (tests already frozen)
- `existing_code` — neighboring types/signatures only

Not a job: `cmd/server` wiring, Swagger, new modules, auth stacks, multi-file type graphs.

## TDD loop (per job)

```text
1. Spec (fixed imports + behavior)
2. Tests exist (Kimi writes the table-driven _test.go)
3. Kimi validates tests
4. Host: freeze test files
5. Sanity: one impl path, allowlist
6. Implement (Gopher, greedy 4096, **not** streamed) → write impl → goimports → **gopls**
   (diagnostics + completion at invented selectors; unique close matches are rewritten on the
   host — `WriteRune` on `io.Writer` becomes `Write`) → allowlist → go test.
   Host injects Neighbor APIs first: exact signatures from same-module packages the
   frozen tests import, then TF-IDF extras (char 3-grams) when the spec names a type
   the import graph missed. No specialist lookup tool. Unused range indexes
   (`for idx, v := range` with `idx` unused) are rewritten to `_` on the host;
   unused accumulators (`buf`, `idx` as a cursor) stay for repair.
7. On fail: Gopher sees error + failing impl only; gopls completions (fallback: `go doc` /
   local types) are appended as a member list; `declared and not used` on a named
   local asks to use it (not `_ = buf`); patch; cap 2–3
   (compile or assertion lines in `*_test.go` mean the impl is wrong — keep repairing;
   do not edit tests)
8. Summary to Kimi (no source dump)
```

Kimi does **not** run `go test`. After freeze, tests stay red until the worker lands.

**Do not** let the same 3B write tests and impl in one unconstrained loop. Freeze the suite first.

If the worker needs different tests, the job **fails** back to Kimi. Do not silently loosen tests.

Kimi implements only after the supervisor returns FAIL (budget exhausted). Do not “try 5 MCP times in the agent thread”.

## MCP surface

| Tool / op | Who calls | Does |
|---|---|---|
| `plan` stays in Kimi | — | not an MCP tool |
| `validate_tests` or Kimi’s own review | Kimi | lock contract |
| `run_job` | Kimi | one impl path, freeze, allowlist, repair loop → **summary** |
| `implement_go` | escape hatch | package-sized writes; `freeze_tests` default true |
| `ask_gopher_kungfu` | Kimi | review/debug existing snippet (`max_tokens` 2048, temp 0.2) |

## Training (when you retrain)

Match what you serve:

- Distill **implement-shaped** prompts (spec + constraints + files + signatures); store shards under `data/library/<topic>/` and `synthetic/topics/`
- Bias **write** and **debug**; debug rows are repair packets (broken file + verbatim `gc`/`go test` line → patched file). Curriculum asks for illegal-Go / Python-ism debug items (`for`/`else`, `elif`, …), `return 0` / `err == 0` / `WriteByte(...) == 0` (`cannot convert 0 … to type interface{Error() string}`), `w.Write(r)` with a rune (`[]byte(string(r))` / `io.WriteString`), `seen := map[string]int{}` then assigned `map[string][]string` (declare the stored value type), and `declared and not used` (use the named local; `_` only for `for _, v := range`). Write rows include `flag.FlagSet`: `Parse(args)` when argv is already sliced, plus `--output`/`-o` and `-f`. The host blanks an unused **range index** only; accumulators like `buf` stay so the specialist can finish the algorithm. Catalog topic **Go compiler** is for that class of diagnostics. Re-running distill on a finished project **fills short syllabus items** (new topic or new rows) and only wipes `synthetic/inbox.jsonl` when every current item is already at quota.
- **Compile-gate**: extract `### path.go` fences, `go test` in a stdlib stub module, drop failures (skip the gate only if `go` is missing)
- One-file examples; `seq_len` and `max_tokens` in the same band (**4096** for 3B/4B/7B/Lite; do not train at 2048 and infer at 6144)
- Thinking **off** (no CoT in the cartridge)
- **~400–600** unique green jobs to see if a 3B worker is real; **~1000–1500** unique chores if that A/B wins — not 1500 paraphrases
- Niche: **Go + testing** (or one library)
- Eval: `python -m app.pipeline.eval_chores` — hidden tests, pass@1 and pass@3 — not JSONL loss
- Export writes **Q4_K_M** (card default) plus **Q5_K_M** beside it from the same merge; ship the winner

## Retrain playbook

1. Compile-gate current `synthetic/` — how many rows even build?
2. Redistill **Go + testing** (or one library), implement+repair only, unique jobs.
3. Train **Qwen3 4B** at 4096, thinking off, greedy serve, Q4 vs Q5 on the same merge.
4. Same jsonl on **DeepSeek-Coder-V2 Lite** only if the 4B A/B is real and VRAM is already Lite-class.
5. Judge with `tests/eval_chores/` (add more stdlib chores over time), not train loss.

## Kimi still does

Layout at a high level, job list, test validation, Swagger, `main`/middleware, `go.mod` for **new** modules, anything off the allowlist.

## Status

1. Host: parse fences → write → goimports → gopls fix → `go test` → 2–3 farm retries → summary — **done**
2. Freeze tests by default — **done**
3. `run_job` supervisor — **done**
4. Kilo rule: do not `write` worker source; implement yourself only on FAIL — **done**
5. Cartridge allowlist on `card.json` + post-write check — **done**
6. Compile-gated distill + 4096 train defaults + dual quant + eval_chores harness — **done** (run a real retrain separately)
