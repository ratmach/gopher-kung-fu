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

## Layout

```
data/projects/<slug>/     # project.json, curriculum, synthetic/*.jsonl, runs/
data/secrets.json         # API keys (gitignored)
cartridges/<slug>/        # GGUF + card.json
```

## Later (not built)

Repo ingest, Docker CUDA, agent tools / subagent protocol, peer SLM handoff, cloud GPUs.
