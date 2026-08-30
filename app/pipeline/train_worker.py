from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"


def log(message: str) -> None:
    print(message, flush=True)


def load_sharegpt(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        conversations = []
        for turn in data.get("conversations", []):
            role = turn.get("from") or turn.get("role")
            if role not in {"human", "gpt"}:
                continue
            conversations.append({"from": role, "value": turn.get("value", "")})
        if len(conversations) >= 2:
            rows.append({"conversations": conversations})
    return rows


def formatting_prompts_func(examples, tokenizer):
    texts = []
    for conversations in examples["conversations"]:
        messages = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        texts.append(text)
    return {"text": texts}


def eval_run_plan(*, has_eval: bool, eval_steps: int) -> dict[str, Any]:
    if not has_eval:
        return {"eval_strategy": "no"}
    if int(eval_steps) > 0:
        return {"eval_strategy": "steps", "eval_steps": int(eval_steps)}
    return {"eval_strategy": "epoch"}


def sft_config_kwargs(cfg: dict[str, Any], *, has_eval: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(Path(cfg["adapter_dir"]) / "trainer"),
        "per_device_train_batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": int(cfg["grad_accum"]),
        "num_train_epochs": float(cfg["epochs"]),
        "learning_rate": float(cfg["learning_rate"]),
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": "cosine",
        "dataset_text_field": "text",
        "max_seq_length": int(cfg["seq_len"]),
        "packing": False,
        "report_to": "none",
        "seed": 42,
        **eval_run_plan(has_eval=has_eval, eval_steps=int(cfg.get("eval_steps") or 0)),
    }
    if has_eval:
        kwargs["per_device_eval_batch_size"] = int(cfg["batch_size"])
    return kwargs


def build_sft_config(cfg: dict[str, Any], *, has_eval: bool):
    from trl import SFTConfig

    kwargs = sft_config_kwargs(cfg, has_eval=has_eval)
    try:
        return SFTConfig(**kwargs)
    except TypeError:
        if "eval_strategy" not in kwargs:
            raise
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return SFTConfig(**kwargs)


def maybe_mask_prompts(
    trainer,
    *,
    enabled: bool,
    instruction_part: str | None = None,
    response_part: str | None = None,
):
    if not enabled:
        return trainer
    try:
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        log("train_on_responses_only not in this Unsloth; training on full sequences")
        return trainer
    fallback_instruction = instruction_part or QWEN_INSTRUCTION_PART
    fallback_response = response_part or QWEN_RESPONSE_PART
    try:
        wrapped = train_on_responses_only(trainer)
        log("Loss on assistant turns only")
        return wrapped
    except TypeError:
        try:
            wrapped = train_on_responses_only(
                trainer,
                instruction_part=fallback_instruction,
                response_part=fallback_response,
            )
            log("Loss on assistant turns only (explicit chat markers)")
            return wrapped
        except Exception as exc:
            log(f"train_on_responses_only failed ({exc}); training on full sequences")
            return trainer
    except Exception as exc:
        log(f"train_on_responses_only failed ({exc}); training on full sequences")
        return trainer


def run(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    train_path = Path(cfg["train_jsonl"])
    if not train_path.exists():
        raise SystemExit(f"missing train jsonl: {train_path}")
    rows = load_sharegpt(train_path)
    if len(rows) < 4:
        raise SystemExit(f"need at least 4 train examples, found {len(rows)}")

    log(f"Loading Unsloth model {cfg['train_id']} (seq={cfg['seq_len']})…")
    from app.pipeline.gpu import require_cuda, unsloth_worker_env

    gpu = require_cuda()
    log(f"Using {gpu.get('device_name')} with {gpu.get('torch')} (CUDA {gpu.get('cuda_built')})")
    unsloth_worker_env()
    log("Importing Unsloth — first run compiles kernels and can look idle for a minute.")
    try:
        if cfg.get("vision"):
            from unsloth import FastLanguageModel, FastVisionModel
        else:
            from unsloth import FastLanguageModel

            FastVisionModel = None  # type: ignore[assignment, misc]
    except ImportError as exc:
        raise SystemExit(
            "Unsloth is not installed. After CUDA PyTorch works, run "
            "`pip install unsloth datasets` (see scripts/install_train.ps1)."
        ) from exc
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for training. pip install datasets") from exc

    max_seq = int(cfg["seq_len"])
    load_kwargs: dict = {
        "load_in_4bit": True,
        "max_seq_length": max_seq,
    }
    if cfg.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True
    if cfg.get("vision"):
        model, tokenizer = FastVisionModel.from_pretrained(
            cfg["train_id"],
            **load_kwargs,
        )
        model = FastVisionModel.get_peft_model(
            model,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=0,
            bias="none",
            target_modules=None,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth",
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg["train_id"],
            **load_kwargs,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    log(f"Dataset: {len(dataset)} examples")

    from trl import SFTTrainer

    eval_path = Path(cfg["eval_jsonl"])
    eval_dataset = None
    if eval_path.exists() and eval_path.stat().st_size > 0:
        eval_rows = load_sharegpt(eval_path)
        if eval_rows:
            eval_dataset = Dataset.from_list(eval_rows).map(
                lambda batch: formatting_prompts_func(batch, tokenizer),
                batched=True,
                remove_columns=["conversations"] if "conversations" in eval_rows[0] else None,
            )

    args = build_sft_config(cfg, has_eval=eval_dataset is not None)

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "args": args,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)

    trainer = maybe_mask_prompts(
        trainer,
        enabled=bool(cfg.get("train_on_responses_only", True)),
        instruction_part=cfg.get("instruction_part"),
        response_part=cfg.get("response_part"),
    )

    log("QLoRA training started")
    trainer.train()
    adapter_dir = Path(cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log(f"Saved LoRA adapter to {adapter_dir}")

    merged_dir = Path(cfg["merged_dir"])
    merged_dir.mkdir(parents=True, exist_ok=True)
    log("Merging LoRA into fp16 weights…")
    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as exc:
        log(f"merged_16bit failed ({exc}); saving full model instead")
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
    log(f"Merged fp16 weights at {merged_dir}")
    log("TRAIN_DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth QLoRA worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        run(Path(args.config))
    except KeyboardInterrupt:
        log("TRAIN_ERROR: interrupted (do not Ctrl+C the API while Unsloth is compiling)")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as exc:
        log(f"TRAIN_ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
