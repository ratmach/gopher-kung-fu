# Specialist farm

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
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
cd web
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

**Training** needs **WSL2 + NVIDIA CUDA**. Native Windows PyTorch/Unsloth is out of scope.

```bash
# inside WSL
pip install -e .
pip install unsloth datasets
# first run will pull unsloth/Qwen3-1.7B-unsloth-bnb-4bit
```

VRAM: ~8 GB for Qwen3 1.7B QLoRA; prefer 12 GB+ for Ministral 3B.

**Export** uses Unsloth `save_pretrained_gguf` when available, otherwise llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize`). Set `LLAMA_CPP_DIR` if you use the latter.

## Farm server

```bash
go run ./server/cmd/cartridge-server -cartridges ./cartridges -addr 127.0.0.1:8080
```

Needs `llama-server` on `PATH` (or `--llama-server` / `LLAMA_SERVER`) to actually answer. Listing models works from `card.json` alone.

At most `--max-loaded` GGUFs stay in VRAM (default 2). Idle specialists unload (LRU).

## Layout

```
data/projects/<slug>/     # project.json, curriculum, synthetic/*.jsonl, runs/
data/secrets.json         # API keys (gitignored)
cartridges/<slug>/        # GGUF + card.json
```

## Later (not built)

Repo ingest, Docker CUDA, agent tools / subagent protocol, peer SLM handoff, cloud GPUs.
