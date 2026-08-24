from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)


def find_tool(names: list[str]) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    extra = os.environ.get("LLAMA_CPP_DIR")
    if extra:
        root = Path(extra)
        for name in names:
            for candidate in (root / name, root / "build" / "bin" / name):
                if candidate.exists():
                    return str(candidate)
    return None


def convert_with_unsloth(merged_dir: Path, out_path: Path) -> bool:
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        return False
    log("Trying Unsloth save_pretrained_gguf…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(merged_dir),
        load_in_4bit=False,
    )
    tmp = out_path.parent / "_gguf_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method="q4_k_m")
    except TypeError:
        model.save_pretrained_gguf(str(tmp), tokenizer)
    found = list(tmp.glob("*.gguf"))
    if not found:
        return False
    shutil.move(str(found[0]), str(out_path))
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def convert_with_llamacpp(merged_dir: Path, out_path: Path) -> None:
    convert = find_tool(["convert_hf_to_gguf.py"])
    if not convert:
        script_candidates = []
        extra = os.environ.get("LLAMA_CPP_DIR")
        if extra:
            script_candidates.append(Path(extra) / "convert_hf_to_gguf.py")
        for candidate in script_candidates:
            if candidate.exists():
                convert = str(candidate)
                break
    if not convert:
        raise SystemExit(
            "Need Unsloth GGUF export or llama.cpp convert_hf_to_gguf.py. "
            "Set LLAMA_CPP_DIR to a llama.cpp checkout."
        )
    f16 = out_path.with_name(out_path.name.replace(".Q4_K_M.gguf", ".f16.gguf"))
    log(f"Converting HF → GGUF via {convert}")
    subprocess.check_call(
        [sys.executable, convert, str(merged_dir), "--outfile", str(f16), "--outtype", "f16"]
    )
    quant = find_tool(["llama-quantize", "quantize"])
    if not quant:
        raise SystemExit("llama-quantize not found. Build llama.cpp or set LLAMA_CPP_DIR.")
    log("Quantizing to Q4_K_M")
    subprocess.check_call([quant, str(f16), str(out_path), "Q4_K_M"])
    if f16.exists() and f16 != out_path:
        f16.unlink()


def run(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    merged = Path(cfg["merged_dir"])
    if not merged.exists():
        raise SystemExit(f"merged weights missing: {merged}. Train first.")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["gguf_name"]
    log(f"Exporting cartridge {cfg['slug']} → {out_path}")
    if not convert_with_unsloth(merged, out_path):
        convert_with_llamacpp(merged, out_path)
    if not out_path.exists():
        raise SystemExit("GGUF was not produced")
    log(f"Wrote {out_path}")
    log("EXPORT_DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description="GGUF export worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        run(Path(args.config))
    except SystemExit:
        raise
    except Exception as exc:
        log(f"EXPORT_ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
