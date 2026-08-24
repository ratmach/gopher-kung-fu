from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.pipeline.gpu import unsloth_worker_env
from app.pipeline.llama_cpp_tools import ensure_convert_script, ensure_quantize


def log(message: str) -> None:
    print(message, flush=True)


def convert_with_unsloth(merged_dir: Path, out_path: Path) -> bool:
    try:
        unsloth_worker_env()
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
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"Unsloth GGUF failed ({exc}). Falling back to llama.cpp binaries.")
        return False


def convert_with_llamacpp(merged_dir: Path, out_path: Path) -> None:
    convert = ensure_convert_script()
    quant = ensure_quantize()
    f16 = out_path.with_name(out_path.name.replace(".Q4_K_M.gguf", ".f16.gguf"))
    log(f"Converting HF → GGUF via {convert}")
    subprocess.check_call(
        [sys.executable, str(convert), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"],
        cwd=str(convert.parent),
    )
    log(f"Quantizing to Q4_K_M with {quant}")
    subprocess.check_call([str(quant), str(f16), str(out_path), "Q4_K_M"])
    if f16.exists() and f16 != out_path:
        f16.unlink()


def run(config_path: Path) -> None:
    unsloth_worker_env()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    merged = Path(cfg["merged_dir"])
    if not merged.exists():
        raise SystemExit(f"merged weights missing: {merged}. Train first.")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["gguf_name"]
    log(f"Exporting cartridge {cfg['slug']} → {out_path}")
    cmake = shutil.which("cmake")
    if cmake:
        log(f"CMake on PATH: {cmake}")
    # Unsloth's Windows llama.cpp zip includes a nested Svelte UI that exceeds MAX_PATH,
    # then it tries winget install Kitware.CMake even when CMake is already present.
    if sys.platform == "win32":
        log("Windows: using official llama.cpp CPU binaries (no compile, cmake not required).")
        convert_with_llamacpp(merged, out_path)
    elif not convert_with_unsloth(merged, out_path):
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
