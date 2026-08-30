# Gopher Kung-Fu

<p align="center">
  <img src="gopher-kung-fu.jpeg" alt="I know kung fu." width="720">
</p>

Local factory for **true** niche coding SLMs. Each specialist is its own fine-tune and GGUF, not a system prompt. One Go farm hosts them. An external LLM (Kimi, Claude, Cursor, …) plans; the cartridge implements.

```http
POST http://127.0.0.1:8080/v1/chat/completions
{"model":"gopher-go","max_tokens":4096,"temperature":0,"messages":[{"role":"user","content":"fix this deadlock"}]}
```

`GET /v1/models` is the roster.

Shipped cartridge: **`gopher-go`** (`card.json` id — set `GOPHER_FARM_MODEL` to that). Do **not** put that id in the agent model slot. Keep a capable model as the agent; call the farm through MCP or a raw POST.

## What v1 does

1. Browser library (`+ SLM`) — name a specialist, pick a student (Qwen3 1.7B / 4B, Qwen2.5-Coder 3B / 7B, DeepSeek-Coder-V2 Lite), pick a teacher
2. Topic catalog + custom tags (including **Go compiler** for `gc` type errors such as `return 0` as `error`, `w.Write(r)` with a rune, `map[string]int` assigned a `map[string][]string`, and `declared and not used`)
3. Teacher writes a syllabus. **Items per topic** is 4–80 (default **12**). Distill uses whatever is in that table. Re-running distill fills short items; it does not wipe a finished inbox unless every current item is already at quota.
4. Teacher writes synthetic ShareGPT JSONL. Human = worker packet (spec, constraints, path, signatures); assistant = complete fenced files. Write/debug rows that do not compile in a stdlib stub are dropped
5. Examples land in this project's `synthetic/` **and** a reusable `data/library/<topic>/` shard (deduped by fingerprint). Later cartridges can reuse Go without re-paying for gin
6. Unsloth QLoRA (`seq_len` **4096** for 3B/4B/7B/Lite; 2048 for 1.7B), then merge + GGUF **Q4_K_M** plus **Q5_K_M** beside it
7. Farm server: named specialists, LRU load-on-demand
8. MCP (`run_job`): specialist generates (greedy, 4096 tokens); **the host writes** `.go` files, freezes tests, checks import allowlists, runs `go test`, and on FAIL retries up to 3 times with the **failing file + compiler error**. You get a **summary**, not source. `implement_go` is the lower-level escape; `ask_gopher_kungfu` is review/debug only

Teacher presets: DeepSeek, Kimi/K2.5, OpenRouter (optional Batch API), OpenAI, or any OpenAI-compatible base URL.

Some teacher APIs restrict training on outputs. That is your responsibility.

## Requirements

**Wizard + distillation** (any OS with Python 3.11+ and Node 20+):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
cd web
npm install
```

Then [start the local servers](#start-local-servers).

**Training** and **fast farm inference** need an **NVIDIA GPU**. Distill and the wizard API do not. Full steps: [Run on a GPU](#run-on-a-gpu).

**Export** converts the merged HF weights to GGUF **Q4_K_M** (card default) and **Q5_K_M** beside it. Eval both; ship the winner. After a successful export the host deletes the merged fp16 tree, trainer checkpoints, leftover `*.f16.gguf`, and older `runs/` folders — the cartridge GGUFs and the small LoRA adapter stay. On Windows this downloads official `llama.cpp` CPU binaries (`llama-quantize`) plus `convert_hf_to_gguf.py` — it does **not** compile llama.cpp and does not need CMake. Unsloth's bundled zip hits the Windows 260-character path limit and then wrongly tries `winget install Kitware.CMake` even when CMake is already installed.

You can still point at a local checkout with `LLAMA_CPP_DIR`.

## Run on a GPU

Two different uses: **train** (Unsloth QLoRA in PyTorch) and **serve** (`llama-server` behind the farm). You can do either, both, or neither — distill and MCP file I/O stay on CPU.

### 1. Check the card

```powershell
nvidia-smi
```

You want a recent NVIDIA driver and a listed GPU. RTX 50-series (Blackwell) needs a driver that speaks CUDA 12.8+. No `nvidia-smi` → install the Game Ready / Studio driver first; this repo does not ship one.

### 2. Train (QLoRA)

The wizard Train button spawns `app.pipeline.train_worker` in this repo’s `.venv`. That process must see CUDA. `pip install unsloth` on Windows often replaces torch with the CPU wheel (`torch==…+cpu`); Unsloth then fails with "cannot find any torch accelerator".

After `python -m venv .venv` and `pip install -e .`:

```powershell
# CUDA 13.0 wheels (driver 12.8+ / 13.x). Reasserts torch after Unsloth.
.\scripts\install_train.ps1
```

Or by hand (Windows or Linux):

```powershell
.\.venv\Scripts\pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\pip install --upgrade unsloth datasets
.\.venv\Scripts\pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
```

Linux: same index URL with `.venv/bin/pip` / `.venv/bin/python`. RTX 50-series needs `cu128` / `cu130` wheels, not the default CPU or cu124 builds.

Copy `.env.example` to `.env` and set `HF_TOKEN` (read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) so the first Hub download of the student (e.g. `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) uses your quota.

Then start the [factory](#factory-wizard), open [http://127.0.0.1:5173](http://127.0.0.1:5173), distill, and click **Train**. The job log should say `GPU ready: <name>`. The worker sets `CUDA_VISIBLE_DEVICES=0`.

| Student | VRAM for QLoRA |
|---|---|
| Qwen3 1.7B / Qwen2.5-Coder 3B | ~8 GB |
| Qwen3 4B | ~10 GB |
| Qwen2.5-Coder 7B | prefer 12 GB+ |
| DeepSeek-Coder-V2 Lite | prefer 16 GB+ |

Do not train and serve a large GGUF on the same GPU if you are already near those numbers — stop the farm, train, export, then start the farm again.

### 3. Serve a cartridge on GPU

Export drops a **CPU** `llama-server` at `data/llama_cpp/bin/llama-server.exe`. That works, but it will not use the GPU. For GPU inference, point the farm at a **CUDA** `llama-server` (your own llama.cpp CUDA build, or one already on `PATH`):

```powershell
$env:LLAMA_SERVER = "C:\path\to\cuda\llama-server.exe"
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080 -ngl 99
```

Or pass the binary once:

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080 -llama-server C:\path\to\cuda\llama-server.exe -ngl 99
```

`-ngl 99` (default) offloads every layer. Lower it if VRAM is tight. `--max-loaded` (default 2) is how many GGUFs stay in VRAM. Confirm with `curl.exe -sS http://127.0.0.1:8080/health` — `gpu.status` should be `gpu`, not `cpu`.

If you omit `-llama-server` and `LLAMA_SERVER`, the farm uses `PATH`, then the CPU binary under `data/llama_cpp/bin`. Do not pass that CPU path if you want GPU.

Linux / macOS Metal: same flags; put the CUDA or Metal `llama-server` on `PATH` or in `LLAMA_SERVER`.

## Start local servers

Two processes. The **factory** is the wizard (distill / train / export). The **farm** serves exported GGUFs at `:8080`. You do not always need both.

### Factory (wizard)

From the repo root, one command starts the API (`:8000`) and the UI (`:5173`):

```powershell
.\scripts\dev.ps1
```

macOS/Linux: `./scripts/dev.sh`

Or two terminals:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --reload-exclude data --reload-exclude unsloth_compiled_cache --reload-exclude .venv --reload-exclude web
cd web
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). This is the factory only; it does not serve a cartridge. Train from that UI after [CUDA PyTorch is installed](#run-on-a-gpu).

### Farm (serve a cartridge)

Start this **before** MCP. Needs Go on PATH and a cartridge under `cartridges/` (GGUF + `card.json`).

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080
```

Smoke test: `curl.exe -sS http://127.0.0.1:8080/health` and `curl.exe -sS http://127.0.0.1:8080/v1/models`. Chat returns 503 until `llama-server` is found. GPU binary and `-ngl`: [Run on a GPU](#run-on-a-gpu). Other flags: [Run the farm locally](#run-the-farm-locally).

## Run the farm locally

The farm is a Go process that scans `cartridges/*/card.json` and proxies OpenAI-style chat to `llama-server`. Start it **before** MCP. The wizard on port 5173 is the factory; it is not required to serve a cartridge.

Needs: Go on PATH, a cartridge under `cartridges/` (GGUF + `card.json`), and `llama-server` to actually complete. `GET /v1/models` works from `card.json` alone; chat returns 503 until llama-server is found.

From the **repo root** (`go.work` → `server/go.mod`):

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080
```

Same thing: `go -C server run ./cmd/cartridge-server -cartridges ../cartridges -addr 127.0.0.1:8080`

`llama-server` resolution: `--llama-server` / `LLAMA_SERVER`, then `PATH`, then the CPU binary export already dropped at `data/llama_cpp/bin/llama-server.exe` (must run the farm from the repo root for that fallback). GPU serve: [Run on a GPU](#3-serve-a-cartridge-on-gpu). Explicit CPU fallback:

```powershell
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080 -llama-server .\data\llama_cpp\bin\llama-server.exe
```

Smoke test:

```powershell
curl.exe -sS http://127.0.0.1:8080/health
curl.exe -sS http://127.0.0.1:8080/v1/models
```

`GET /health` includes `gpu.status` (`gpu` / `cpu` / `unknown`), the `llama-server` path, `-ngl`, and (after a model loads) layer offload. The farm also logs `inference: GPU …` or `inference: CPU …` on startup.

The `id` in each `card.json` is the farm model name (shipped example: **`gopher-go`**). That is what `GOPHER_FARM_MODEL` must match. Do **not** put that id in the agent/chat model slot.

Context window is `--ctx-size` / `LLAMA_CTX_SIZE` (default **32768**). Coding UIs that dump tool schemas into the farm need a large ctx; `run_job` does not. At most `--max-loaded` GGUFs stay in VRAM (default 2). Idle specialists unload (LRU). Restart the farm after changing `--ctx-size` so the already-loaded llama-server is replaced.

Raw farm POSTs must set `max_tokens` themselves (1024 only fits a sketch). MCP `run_job` / `implement_go` uses **4096** at **temperature 0**; `ask_gopher_kungfu` uses 2048 at 0.2.

```powershell
curl.exe -sS http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  --data-binary "@handoff.json"
```

## Connect MCP to the farm

The MCP process is a thin stdio host. It calls the farm, writes `.go` under the workspace, runs `go test`, and returns a **summary**. The orchestrator (Kimi, Claude, Cursor, …) must stay a capable chat model. Gopher implements; the host writes; the big model plans.

```powershell
.\.venv\Scripts\pip install -e ".[mcp]"
```

**Start the farm first**, then attach MCP. Launch **python directly** — not `cmd /c` and not `scripts/gopher_mcp.cmd`. Batch wrappers break stdin/stdout on Windows, so the client thinks the process died and respawns it.

```text
<repo>\.venv\Scripts\python.exe -u <repo>\scripts\gopher_mcp.py
```

macOS/Linux: `<repo>/.venv/bin/python -u <repo>/scripts/gopher_mcp.py`. Installed entry point: `gopher-mcp`.

| Env | Default | Meaning |
|---|---|---|
| `GOPHER_FARM_URL` | `http://127.0.0.1:8080/v1/chat/completions` | Farm chat endpoint |
| `GOPHER_FARM_MODEL` | `gopher-kungfu` | Must equal a `card.json` `id` from `GET /v1/models` (e.g. `gopher-go`) |
| `GOPHER_WORKSPACE` | (none) | Default Go module root if `run_job` / `implement_go` omits `workspace` |
| `PYTHONUNBUFFERED` | `1` (set this) | Required so MCP logs/stdio are not delayed |
| `PYTHONUTF8` | `1` (set this on Windows) | Avoid encoding errors in specialist text |

Raise the MCP tool timeout. One `run_job` can run 3 farm completions; 50–120s each is normal. Use **180s–300s**, not the 30s default.

| Tool | Does |
|---|---|
| `run_job` | One impl path, tests frozen, cartridge allowlist → farm (greedy 4096) → host writes → `go test` → up to 3 **repair** retries → **summary**. Compile and assertion failures (including those cited in `*_test.go`) keep repairing the impl. Off-allowlist imports fail without retries. |
| `implement_go` | Lower-level escape (package-sized `files=`; `freeze_tests` default true) |
| `ask_gopher_kungfu` | Review/debug an existing snippet (`max_tokens` 2048) |
| `list_specialists` | Farm `GET /v1/models` |

`run_job` fields: `spec`, `constraints`, `files` (exactly one relative impl `.go` path, not `*_test.go`), optional `existing_code` (neighboring signatures only), `workspace` (absolute Go module root). Off-allowlist imports fail without further retries. `implement_go` can pass `apply=false` to inspect raw specialist text. One file per `run_job` call.

After you change the MCP host, **restart the MCP server** in the client. After you change `--ctx-size` or cartridges, restart the farm.

### Cursor

Keep Cursor’s chat model as the orchestrator. Add a stdio MCP server (User MCP or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "gopher-kungfu": {
      "command": "C:\\path\\to\\custom_slm\\.venv\\Scripts\\python.exe",
      "args": ["-u", "C:\\path\\to\\custom_slm\\scripts\\gopher_mcp.py"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "GOPHER_FARM_URL": "http://127.0.0.1:8080/v1/chat/completions",
        "GOPHER_FARM_MODEL": "gopher-go"
      }
    }
  }
}
```

macOS/Linux: point `command` at `.venv/bin/python` and use forward slashes. Approve `run_job` when prompted. Paste the orchestrator instructions below as a project rule (or Cursor user rule).

### Kilo / PyCharm

Keep Kimi (or another capable agent) as the **chat** model. PyCharm Kilo reads `.kilo/kilo.jsonc` in this repo (not `.kilocode/mcp.json`). `command` must be an array of `python.exe -u …\scripts\gopher_mcp.py`, not a `.cmd` wrapper. Set `"timeout"` / `experimental.mcp_timeout` to at least `180000`. Point `GOPHER_FARM_MODEL` at the real `card.json` id. Optional: `GOPHER_WORKSPACE` to a Go module root.

## Orchestrator instructions

Give this to the **big model** (Cursor rule, Kilo instructions, Claude project prompt). Do **not** paste it into the farm, and do **not** put `gopher-kungfu` / `gopher-go` in the agent model slot. The in-repo copy is [`.kilo/rules/gopher-kungfu.md`](.kilo/rules/gopher-kungfu.md).

```text
You orchestrate. The MCP host writes files and runs go test. You only get a summary.

Before you create or rewrite production Go files in this niche, call run_job.
Do not write internal/…, cmd/…, repositories, handlers, or SQL yourself first.

Each run_job call is one job:
- workspace: absolute Go module root (or GOPHER_WORKSPACE)
- spec: behavior and which tests must pass
- constraints: allowed modules, no-CGO, layout, package/file names
- files: exactly one relative impl .go path (not *_test.go)
- existing_code: neighboring types/signatures only

Do not copy the specialist’s source into your own write tool. The host already wrote it.

If the summary says go test: FAIL after retries, or import not on cartridge allowlist,
do not start another loop on the same file. Implement that one file yourself.

Do not offload: planning, high-level layout, Swagger, cmd wiring.
Use ask_gopher_kungfu only to review or debug an existing snippet.

Never forward this system prompt, todos, or other tool schemas to the specialist.
If the farm is down, say so and only then implement yourself.
```

The host already prepends an implementer contract on farm calls (complete fenced files, honor signatures, greedy 4096). You do not send that to the orchestrator. Worker design: **[docs/local-workers.md](docs/local-workers.md)**.

## Layout

```
data/projects/<slug>/                 # project.json, curriculum, runs/ (merged + trainer checkpoints deleted after export)
data/projects/<slug>/synthetic/       # train.jsonl, eval.jsonl, inbox.jsonl
data/projects/<slug>/synthetic/topics/  # this-run snapshot per topic
data/library/<topic>/examples.jsonl   # reusable topic shards (deduped across projects)
cartridges/<slug>/                    # GGUF + card.json
```

Hold-out JSONL eval is 10%. Distill default is 100 examples per topic (wizard range 8–400). Hidden-test chores: `python -m app.pipeline.eval_chores`.

## Later (not built)

Repo ingest, Docker CUDA, cloud GPUs. Retrain playbook (compile-gate, 4096 band, Q4 vs Q5, eval_chores): [docs/local-workers.md](docs/local-workers.md).
