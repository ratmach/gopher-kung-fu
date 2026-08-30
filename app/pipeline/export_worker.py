from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.pipeline.gpu import unsloth_worker_env
from app.pipeline.llama_cpp_tools import ensure_convert_script, ensure_quantize

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M"]


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


def convert_hf_to_f16(merged_dir: Path, f16: Path) -> None:
    convert = ensure_convert_script()
    log(f"Converting HF → GGUF via {convert}")
    subprocess.check_call(
        [sys.executable, str(convert), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"],
        cwd=str(convert.parent),
    )


def quantize_gguf(f16: Path, out_path: Path, quant: str) -> None:
    binary = ensure_quantize()
    log(f"Quantizing to {quant} with {binary}")
    subprocess.check_call([str(binary), str(f16), str(out_path), quant])


def convert_with_llamacpp(merged_dir: Path, out_dir: Path, slug: str, quants: list[str]) -> None:
    f16 = out_dir / f"{slug}.f16.gguf"
    convert_hf_to_f16(merged_dir, f16)
    try:
        for quant in quants:
            quantize_gguf(f16, out_dir / f"{slug}.{quant}.gguf", quant)
    finally:
        if f16.exists():
            f16.unlink()


def run(config_path: Path) -> None:
    unsloth_worker_env()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    merged = Path(cfg["merged_dir"])
    if not merged.exists():
        raise SystemExit(f"merged weights missing: {merged}. Train first.")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = str(cfg.get("slug") or "cartridge")
    quants = [str(item) for item in (cfg.get("quants") or DEFAULT_QUANTS)]
    if not quants:
        quants = list(DEFAULT_QUANTS)
    primary = cfg.get("gguf_name") or f"{slug}.{quants[0]}.gguf"
    log(f"Exporting cartridge {slug} → {', '.join(quants)}")
    cmake = shutil.which("cmake")
    if cmake:
        log(f"CMake on PATH: {cmake}")
    if sys.platform == "win32":
        log("Windows: using official llama.cpp CPU binaries (no compile, cmake not required).")
        convert_with_llamacpp(merged, out_dir, slug, quants)
    elif len(quants) == 1 and quants[0].upper() == "Q4_K_M":
        out_path = out_dir / primary
        if not convert_with_unsloth(merged, out_path):
            convert_with_llamacpp(merged, out_dir, slug, quants)
    else:
        convert_with_llamacpp(merged, out_dir, slug, quants)
    missing = [q for q in quants if not (out_dir / f"{slug}.{q}.gguf").exists()]
    if missing:
        raise SystemExit("GGUF was not produced: " + ", ".join(missing))
    log(f"Wrote {(out_dir / primary)}")
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
