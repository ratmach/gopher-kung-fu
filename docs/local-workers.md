# Local workers

Kimi plans and locks tests; the **MCP host** writes files, runs `go test`, and retries the SLM. Kimi only sees a short summary.

**Built now:** `implement_go` parses `### path.go` fences, writes under `workspace` / `GOPHER_WORKSPACE`, runs `go test` on those packages, retries the specialist up to 3 times on FAIL, returns a summary (not source). The SLM does not emit tool JSON.

**Still planned:** frozen import allowlist on the cartridge, supervisor fan-out, tests generated from plan JSON.

## Why this shape

A 1.7B Q4 cartridge cannot land a compiling package (wrong imports, invented types). Kimi K2.7 Coder already does a feature for about **$1.47**; offloading whole services does not beat that.

What can work:

- **3B-class** student (e.g. Ministral 3B), not 1.7B, for easy–medium **day-to-day** Go
- **One cartridge = one frozen import set** (`gopher-go-redis`, `gopher-go-kafka`, …)
- **One job = one file or one function**, signatures in `existing_code`
- **`go test` is the judge**, not Kimi’s opinion of the source
- Kimi output = **spec / tests / summary**, never the worker’s implementation

LeetCode-hard is out of scope. Gin+SQL+JWT in one model is out of scope.

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
  call supervisor once per job
  on FAIL after budget: implement yourself
        │
        ▼
Supervisor (code, not an LLM)
  sanity checks, freeze tests, retry budget, timeout
  fan-out jobs; never forward drafts to Kimi
        │
        ▼
MCP host (this process)
  parse ### path / ```go fences
  write under workspace allowlist
  go test
  feed compiler/test errors back to Gopher (2–3×)
  return summary only
        │
        ▼
Farm  POST /v1/chat/completions
  Gopher GGUF: text in → Go out
```

**Supervisor** is a state machine in the MCP server (or a thin module it calls). Not a second model “managing” workers.

## Cartridge = frozen stack

At export / MCP start, a cartridge declares an **allowlist** (and a denylist). Example:

```text
allowed:  stdlib, github.com/redis/go-redis/v9
forbidden: github.com/go-redis/redis (v8), CGO
```

Kimi’s plan may only name those imports. If the task needs another library, **do not offload**.

Setup probes (no per-OS file-writer zoo):

- `go version` on PATH
- workspace root
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
2. Tests exist
     Kimi writes the table-driven _test.go  (preferred: small, high-leverage output)
     or worker drafts tests once, then Kimi reviews
3. Kimi validates tests
     PASS + freeze, or FAIL + what to change
     checklist: cases match spec, not tautological, allowlisted imports, job-sized
4. Host: freeze test files (worker must not edit them)
5. Sanity (code): paths, allowlist, job size
6. Implement (Gopher) → write impl → go test
7. On fail: Gopher sees error + failing impl only; patch impl; repeat
     cap 2–3 local retries (not 5 round-trips through Kimi)
8. Summary to Kimi
     wrote internal/foo.go; go test PASS
     or FAIL after N; compiler/test error (no source dump)
```

Kimi does **not** run `go test`. After freeze, tests stay red until the worker lands.

**Do not** let the same 3B write tests and impl in one unconstrained loop — it will make both match and go green. Freeze the suite first.

If the worker needs different tests, the job **fails** back to Kimi. Do not silently loosen tests.

Kimi implements only after the supervisor returns FAIL (budget exhausted). Do not “try 5 MCP times in the agent thread” — each miss pollutes context, then Kimi writes anyway.

## MCP surface (target)

Keep Gopher text-in / Go-out. Host tools (not for Kimi to pass file bodies):

| Tool / op | Who calls | Does |
|---|---|---|
| `plan` stays in Kimi | — | not an MCP tool |
| `validate_tests` or Kimi’s own review | Kimi | lock contract; optional if Kimi wrote tests in-repo |
| `run_job` / supervisor entry | Kimi | spec + frozen test paths + impl path → loop → **summary** |
| (internal) write, `go test`, consult farm | host | never exposed as “here, paste contents” |

`ask_gopher_kungfu` can stay for snippet review; it is not the worker path.

Today: `implement_go` + `ask_gopher_kungfu` in `app/mcp_server.py`. Replace the “return source, Kimi writes” contract with `run_job` → summary.

## Training (when you retrain)

Match what you serve:

- Distill **implement-shaped** prompts (contract + spec + constraints + files + `existing_code`); store shards under `data/library/<topic>/` and `synthetic/topics/`
- Pin the same stack in curriculum + distill
- **Compile-gate**: extract Go, `go test` in a stub module, drop failures
- One-file examples; `seq_len` and `max_tokens` in the same band (do not train at 2048 and infer at 6144)
- Repair examples: almost-right file + real `go test` error → patch
- **~400–600** unique green jobs to see if a 3B worker is real; **~1000–1500** unique chores if that A/B wins — not 1500 paraphrases
- Niche: **Go + testing** (or one library), not gin+SQL+concurrency in one cartridge
- Eval: 30–50 held-out chores with **hidden** tests; pass@1 and pass@3 after local repair — not JSONL loss

## Kimi still does

Layout at a high level, job list, test validation, Swagger, `main`/middleware, `go.mod` for **new** modules, anything off the allowlist.

## Implement in this order

1. Host: parse fences → write under allowlist → `go test` → 2–3 farm retries → summary string
2. Freeze tests (worker cannot overwrite `*_test.go` for that job)
3. Supervisor state machine + `run_job` MCP tool
4. Kilo rule: do not `write` worker source; implement yourself only on FAIL
5. Cartridge allowlist on `card.json` + sanity
6. Retrain 3B Go+testing with compile-gate (separate from wiring the host)
