# Gopher Kung-Fu 

## Specialist farm

Local factory for **true** niche coding SLMs. Each specialist is its own fine-tune and GGUF, not a system prompt. One Go server hosts all of them. An external LLM (Cursor, Claude, …) asks by name:

```http
POST http://127.0.0.1:8080/v1/chat/completions
{"model":"gopher-kungfu","messages":[{"role":"user","content":"fix this deadlock"}]}
```

`GET /v1/models` is the roster.

## What v1 does

1. Browser library (`+ SLM`)
2. Topic catalog + custom tags
3. Teacher API writes a syllabus, then synthetic coding data (ShareGPT JSONL on disk)
4. Unsloth QLoRA on **Qwen3 1.7B** (or Ministral 3B)
5. Merge + GGUF **Q4_K_M**
6. Farm server: many named specialists, LRU load-on-demand

Teacher presets: DeepSeek, Kimi/K2.5, OpenRouter, OpenAI, or any OpenAI-compatible base URL.

Some teacher APIs restrict training on outputs. That is your responsibility.

## Requirements

**Wizard + distillation** (any OS with Python 3.11+ and Node 20+):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
cd web
npm install
```

Two terminals:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --reload-exclude data --reload-exclude unsloth_compiled_cache --reload-exclude .venv --reload-exclude web
cd web
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

**Training** needs an **NVIDIA GPU** and a **CUDA PyTorch** build. `pip install unsloth` on Windows often pulls the CPU wheel (`torch==…+cpu`). Unsloth then fails with "cannot find any torch accelerator".

```powershell
# in this repo's .venv — installs CUDA 13.0 wheels (driver 12.8+ / 13.x)
.\scripts\install_train.ps1
```

Or by hand:

```powershell
.\.venv\Scripts\pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\pip install --upgrade unsloth datasets
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

RTX 50-series (Blackwell) needs CUDA 12.8+ wheels (`cu128` / `cu130`), not the default CPU or cu124 builds.

VRAM: ~8 GB for Qwen3 1.7B QLoRA; prefer 12 GB+ for Ministral 3B. First train downloads `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`.

VRAM: ~8 GB for Qwen3 1.7B QLoRA; prefer 12 GB+ for Ministral 3B.

**Export** converts the merged HF weights to GGUF **Q4_K_M**. On Windows this downloads official `llama.cpp` CPU binaries (`llama-quantize`) plus `convert_hf_to_gguf.py` — it does **not** compile llama.cpp and does not need CMake. Unsloth's bundled zip hits the Windows 260-character path limit and then wrongly tries `winget install Kitware.CMake` even when CMake is already installed.

You can still point at a local checkout with `LLAMA_CPP_DIR`.

## Farm server

From the repo root (uses `go.work` → `server/go.mod`):

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080
```

Or: `go -C server run ./cmd/cartridge-server -cartridges ../cartridges -addr 127.0.0.1:8080`

Needs `llama-server` to actually answer. Listing models works from `card.json` alone.

Resolution order: `--llama-server` / `LLAMA_SERVER`, then `PATH`, then the CPU binary export already dropped at `data/llama_cpp/bin/llama-server.exe` (run from the repo root). Explicit path:

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080 -llama-server .\data\llama_cpp\bin\llama-server.exe
```

Context window is `--ctx-size` / `LLAMA_CTX_SIZE` (default **32768**, Qwen3's trained length). Coding clients like Kilo Code wrap a one-word prompt in a large system prompt + tool schemas; 4096 is too small for that.

At most `--max-loaded` GGUFs stay in VRAM (default 2). Idle specialists unload (LRU). Restart the farm after changing `--ctx-size` so the already-loaded llama-server is replaced.

## Kilo Code (PyCharm MCP handoff)

Do **not** set Kilo’s chat model to `gopher-kungfu`. That dumps the agent system prompt + tools into the 1.7B specialist. Keep **Kimi** as the agent; the cartridge is an MCP tool that POSTs a small `messages: [{role, content}]` body.

PyCharm’s plugin does **not** read `.kilocode/mcp.json` (`mcpServers`). It uses the same `kilo.jsonc` as the CLI:

```powershell
.\.venv\Scripts\pip install -e ".[mcp]"
```

1. Start the farm on `127.0.0.1:8080`.
2. In PyCharm: **Settings → Tools → Kilo Code**. Provider/model = Kimi K2.7, not the farm.
3. Open **MCP Servers** (under Agent Behaviour / MCP Server Settings) and confirm **gopher-kungfu** is enabled. Restart the Kilo tool window if it was already open.
4. If the server **keeps restarting**, the command must be `python.exe -u …\scripts\gopher_mcp.py` — not `cmd /c` and not `gopher_mcp.cmd`. Batch wrappers break stdin/stdout on Windows, so Kilo thinks the process died and respawns it. Also disable any duplicate entry in `.kilocode/mcp.json`.
5. Approve `implement_go` (and `ask_gopher_kungfu`) when prompted. Pass **`workspace`** as the Go module root (or set `GOPHER_WORKSPACE`). The host writes `.go` files and runs `go test`; Kimi should **not** paste the source into a write tool.

Wired in-repo:

- `.kilo/kilo.jsonc` — JetBrains / CLI MCP (`type: local`, `command` array)
- `.kilo/rules/gopher-kungfu.md` — when to call the specialist
- `implement_go` — specialist generates; **MCP host writes** + optional `go test` (up to 3 attempts)
- `ask_gopher_kungfu` — review/debug existing snippets
- `scripts/gopher_mcp.py` — stdio MCP server (launch with `.venv\Scripts\python.exe -u`)

Override farm URL/model with `GOPHER_FARM_URL` / `GOPHER_FARM_MODEL`. Set `GOPHER_WORKSPACE` to the target Go module if you do not want to pass `workspace` on every call. On macOS/Linux use `"command": [".venv/bin/python", "scripts/gopher_mcp.py"]` instead of the `.cmd` wrapper.

## Local workers (host write)

`implement_go` parses `### path.go` fences, writes under the workspace (no `..`, only `.go`), and returns a **summary**. Set `apply=false` only if you need the raw specialist text. `freeze_tests=true` skips `*_test.go`. See **[docs/local-workers.md](docs/local-workers.md)**.

## LLM → SLM handoff (current)

The specialist is the **implementer**. The big model (Kimi, Claude, …) plans, lists files, applies patches, and runs tests. Do not put `gopher-kungfu` in the agent model slot.

**Reply cap** (completion tokens, not the 32k context window):

| Call | `max_tokens` | Use |
|---|---|---|
| `ask_gopher_kungfu` / mode `ask` | **2048** | review / debug a snippet |
| `implement_go` / mode `implement` | **6144** | full files |

A 1024 cap only fits a sketch. MCP sets these automatically. Raw farm POSTs must set `max_tokens` yourself or llama-server will stop mid-file.

### Implement prompt (what MCP prepends)

The farm wraps `implement` calls with this contract, then your spec:

````text
You are the Go specialist IMPLEMENTER, not an advisor.
Return production code. Do not give architecture essays, package shopping lists, or sketches.

For each file, use exactly this shape:

### relative/path.go
```go
package ...
```

Rules:
- Complete files only. No TODOs, no "the rest is similar", no omitted functions.
- Honor constraints exactly (allowed modules, CGO, layout, names).
- Do not add dependencies the spec did not allow.
- If tests are requested, include complete `_test.go` files.
- After the files, at most two sentences. No preamble before the first file.
````

### How the orchestrator should call it

One package or a few files per call — not the whole service.

1. Decide layout and constraints (you / the LLM).
2. Call `implement_go` with `spec`, `constraints`, and `files`.
3. Write the returned `### path` files to disk. Do not re-type the package.
4. `go test`. On failure, call `implement_go` again with the error and the failing file.

MCP tool fields: `spec`, `constraints`, `files` (newline-separated paths), optional `existing_code` (neighboring signatures only).

### Local run (no Kilo)

Farm up, then implement-shaped POST (`max_tokens` 6144):

```powershell
curl.exe -sS http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  --data-binary "@handoff.json"
```

`handoff.json`:

````json
{
  "model": "gopher-kungfu",
  "stream": false,
  "max_tokens": 6144,
  "messages": [
    {
      "role": "user",
      "content": "You are the Go specialist IMPLEMENTER, not an advisor.\nReturn production code.\n\nFor each file:\n### relative/path.go\n```go\npackage ...\n```\n\nSpec:\nHTTP todo API with in-memory SQLite. Tests in-process.\n\nConstraints:\nmodernc.org/sqlite only. No CGO. No github.com/lib/pq.\nLayout: cmd/server, internal/db, internal/todo.\n\nFiles to write:\ninternal/todo/model.go\ninternal/todo/repository.go\ninternal/todo/repository_test.go"
    }
  ]
}
````

Review/debug (cap 2048):

````json
{
  "model": "gopher-kungfu",
  "stream": false,
  "max_tokens": 2048,
  "messages": [
    {
      "role": "user",
      "content": "Fix this deadlock.\n\nExisting code:\n```go\n// paste snippet\n```"
    }
  ]
}
````

Python helper (same caps): `consult(..., mode="implement")` vs `mode="ask"` in `app/farm_consult.py`.

## Layout

```
data/projects/<slug>/                 # project.json, curriculum, runs/
data/projects/<slug>/synthetic/       # train.jsonl, eval.jsonl, inbox.jsonl
data/projects/<slug>/synthetic/topics/  # this-run snapshot per topic
data/library/<topic>/examples.jsonl   # reusable topic shards (deduped across projects)
cartridges/<slug>/                    # GGUF + card.json
```

Distill asks the teacher for **spec-in / code-out** examples (complete fenced files, little prose) and keeps them if the assistant turn is actually code. Each topic is merged into `data/library/<topic>/` so a later cartridge can reuse Go without re-paying for gin. Project `train.jsonl` is still this run’s mix (plus 10% eval).

## Later (not built)

Repo ingest, Docker CUDA, cloud GPUs. Remaining worker pieces (cartridge allowlist, supervisor fan-out): [docs/local-workers.md](docs/local-workers.md).
