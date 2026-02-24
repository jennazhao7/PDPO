#!/usr/bin/env python3
"""
Stage-2 LoRA training (SFT/MLE) on the D2 split.

Loads a base model, attaches and freezes the Stage-1 adapter, then adds a new
Stage-2 adapter and trains only that adapter. No DP flipping is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


class DebugTrainer(Trainer):
    """Prints raw outputs.loss from inside training step for comparison with logged value."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.state.global_step <= 3:
            print(f"[InsideTrainer] step={self.state.global_step} outputs.loss={loss.item():.6f}")
        return (loss, outputs) if return_outputs else loss
from torch.utils.data import DataLoader


# Default: D2 output from rr_stream_flip (partition 1, RR-flipped). Run rr_stream_flip first.
DEFAULT_DATA = "lora/preprocessing/d2_rr_flipped.jsonl"


def load_causal_lm_compat(model_id: str, **kwargs):
    """Load CausalLM with safetensors-first strategy and torch<2.6 fallback."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True, **kwargs)
    except Exception as safe_exc:
        original_import_check = None
        original_modeling_check = None
        try:
            import transformers.utils.import_utils as import_utils
            import transformers.modeling_utils as modeling_utils
            original_import_check = import_utils.check_torch_load_is_safe
            original_modeling_check = modeling_utils.check_torch_load_is_safe
            import_utils.check_torch_load_is_safe = lambda: None
            modeling_utils.check_torch_load_is_safe = lambda: None
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            print(
                f"[Load] WARNING: safetensors unavailable ({safe_exc}); "
                "loaded via torch.load compatibility fallback."
            )
            return model
        finally:
            try:
                if original_import_check is not None:
                    import_utils.check_torch_load_is_safe = original_import_check
                if original_modeling_check is not None:
                    modeling_utils.check_torch_load_is_safe = original_modeling_check
            except Exception:
                pass


def find_first_key(cols: List[str], candidates: List[str]) -> str:
    for key in candidates:
        if key in cols:
            return key
    return ""


def build_text(example: Dict, prompt_key: str, response_key: str) -> str:
    prompt = example.get(prompt_key, "")
    response = example.get(response_key, "")
    return f"{prompt}\n\n{response}"


def read_first_n_jsonl(path: str, n: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


def compute_length_stats(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    mean = sum(lengths) / len(lengths)
    return {"mean": mean, "p50": percentile(lengths, 0.5), "p95": percentile(lengths, 0.95)}


def hash_prompt_response(prompt: str, response: str) -> str:
    raw = f"{prompt}\n{response}".encode("utf-8")
    return str(hash(raw))


def build_generation_inputs(tok, prompts: List[str], device: torch.device) -> Dict:
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tok.model_max_length if tok.model_max_length else 512,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-2 LoRA training (SFT/MLE)")
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument(
        "--stage1_adapter",
        required=True,
        help="Path to Stage-1 LoRA adapter or HF repo (e.g. Jennazhao7/pdpo-lora)",
    )
    ap.add_argument(
        "--stage1_subfolder",
        default=None,
        help="Subfolder in HF repo (e.g. gpt2-large/truthydpo/stage1/eps_1.0)",
    )
    ap.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Full dataset (JSONL); D2 selected via partition args",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir for Stage-2 adapter (default: outputs/stage2_lora_{model}_{stage1}).",
    )
    ap.add_argument(
        "--manifest_out",
        default=None,
        help="Manifest JSON path (default: {out}/M2_manifest.json).",
    )
    ap.add_argument(
        "--output_suffix",
        default="",
        help="Appended to auto-derived --out when --out is not set (e.g. eps1_seed42).",
    )

    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--response_key", default="chosen")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules for stage2",
    )
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit if bitsandbytes is available.",
    )
    ap.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load base model in 8-bit if bitsandbytes is available.",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training (default off to avoid scaler issues).",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training if supported.",
    )
    ap.add_argument(
        "--smoke_test",
        action="store_true",
        default=False,
        help="Run overfit-8 smoke test before full training.",
    )
    ap.add_argument(
        "--no_smoke_test",
        action="store_false",
        dest="smoke_test",
        help="Disable overfit-8 smoke test.",
    )
    ap.add_argument("--smoke_steps", type=int, default=50)
    ap.add_argument(
        "--smoke_lr",
        type=float,
        default=None,
        help="Smoke test LR (defaults to --lr).",
    )
    ap.add_argument(
        "--eval_prompts_json",
        type=str,
        default=None,
        help="Optional JSON list of prompts for generation regression check.",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _path_slug(s: str) -> str:
    """Turn model/adapter path into a safe dir name."""
    return s.replace("/", "--").replace("\\", "--").strip("-")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.out is None:
        model_slug = _path_slug(os.path.basename(args.model.rstrip("/")))
        stage1_slug = _path_slug(os.path.basename(args.stage1_adapter.rstrip("/")))
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        args.out = f"outputs/stage2_lora_{model_slug}_{stage1_slug}{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")

    print(f"[Config] out={args.out} manifest={args.manifest_out}")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)

    # Data sanity checks (before loading model)
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Stage-2 data not found: {args.data}")

    first_rows = read_first_n_jsonl(args.data, 50)
    if not first_rows:
        raise ValueError("Stage-2 data is empty (first 50 lines missing).")

    # Load dataset: D2 output from rr_stream_flip (already partitioned + RR-flipped)
    raw = load_dataset("json", data_files=args.data)["train"]
    cols = raw.column_names
    prompt_key = args.prompt_key if args.prompt_key in cols else find_first_key(cols, ["prompt", "question", "instruction"])
    response_key = args.response_key if args.response_key in cols else find_first_key(cols, ["chosen", "answer", "response"])
    if not prompt_key or not response_key:
        raise ValueError(f"Could not find prompt/response columns in: {cols}")

    print(f"[Data] loaded {len(raw)} rows from {args.data} (D2 from rr_stream_flip)")

    for i, row in enumerate(first_rows):
        if prompt_key not in row or response_key not in row:
            raise ValueError(f"Row {i} missing prompt/response keys.")
        prompt = row.get(prompt_key, "")
        response = row.get(response_key, "")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"Row {i} prompt/response must be strings.")
        if not prompt.strip() or not response.strip():
            raise ValueError(f"Row {i} prompt/response must be non-empty.")
        if len(response.strip()) < 5:
            raise ValueError(f"Row {i} response too short (<5 chars).")

    n_total = len(raw)
    if n_total <= 0:
        raise ValueError("Stage-2 data has zero examples.")

    # OpenLLaMA model card recommends slow tokenizer to avoid tokenization mismatch.
    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    prompt_lens = []
    response_lens = []
    hashes = {}
    for row in raw:
        prompt = row.get(prompt_key, "")
        response = row.get(response_key, "")
        prompt_lens.append(len(tok.encode(prompt, add_special_tokens=False)))
        response_lens.append(len(tok.encode(response, add_special_tokens=False)))
        key = hash_prompt_response(prompt, response)
        hashes[key] = hashes.get(key, 0) + 1

    dup_count = sum(1 for v in hashes.values() if v > 1)
    dup_ratio = dup_count / max(1, len(hashes))
    if dup_ratio > 0.01:
        print(f"⚠️  Duplicate prompt+response hashes >1%: {dup_ratio:.3f}")

    stats = {
        "N": n_total,
        "prompt_len": compute_length_stats(prompt_lens),
        "response_len": compute_length_stats(response_lens),
        "duplicate_ratio": dup_ratio,
    }
    with open(os.path.join(args.out, "stage2_data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    def to_text(example: Dict) -> Dict:
        return {"text": build_text(example, prompt_key, response_key)}

    ds = raw.map(to_text, remove_columns=cols, desc="Formatting text")

    def tokenize(batch: Dict) -> Dict:
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")

    # Device
    cuda_ok = torch.cuda.is_available()
    dtype = torch.float16 if cuda_ok else torch.float32

    # Base model
    use_4bit = args.load_in_4bit
    use_8bit = args.load_in_8bit
    if use_4bit and use_8bit:
        raise ValueError("Use only one of --load_in_4bit or --load_in_8bit.")
    if use_4bit or use_8bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError("bitsandbytes is required for 4-bit/8-bit loading.") from exc

    model = load_causal_lm_compat(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=None,
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
    )
    model.config.use_cache = False
    if hasattr(model.config, "loss_type"):
        model.config.loss_type = "ForCausalLMLoss"

    # Attach Stage-1 adapter and freeze it
    if args.stage1_subfolder:
        model = PeftModel.from_pretrained(
            model, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1"
        )
    else:
        model = PeftModel.from_pretrained(model, args.stage1_adapter, adapter_name="stage1")

    # Add Stage-2 adapter and train only it
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--target_modules must include at least one module.")

    stage2_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.add_adapter("stage2", stage2_config)
    try:
        model.set_adapter(["stage1", "stage2"])
    except Exception:
        model.set_adapter("stage2")

    for name, param in model.named_parameters():
        param.requires_grad = ".stage2." in name

    trainable = 0
    total = 0
    stage1_trainable = 0
    non_lora_trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if ".stage1." in name:
                stage1_trainable += param.numel()
            if "lora_" not in name:
                non_lora_trainable += param.numel()
    print(
        f"[Trainable] trainable={trainable} total={total} "
        f"({100 * trainable / max(1,total):.4f}%)"
    )
    assert trainable > 0, "No trainable parameters found."
    assert stage1_trainable == 0, "Stage-1 adapter has trainable params."
    assert non_lora_trainable == 0, "Non-LoRA params are marked trainable."

    # Adapter state prints
    print(f"[AdapterState] adapters: {getattr(model, 'peft_config', {}).keys()}")
    print(f"[AdapterState] peft_config keys: {getattr(model, 'peft_config', {}).keys()}")
    print(f"[AdapterState] active_adapter: {getattr(model, 'active_adapter', None)}")
    trainable_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[AdapterState] num_trainable: {sum(p.numel() for _, p in trainable_list)}")
    print(f"[AdapterState] sample_trainable_names: {[n for n, _ in trainable_list[:20]]}")

    stage2_trainable = []
    stage2_frozen = []
    for n, p in model.named_parameters():
        if "lora" in n.lower() or "stage2" in n.lower():
            (stage2_trainable if p.requires_grad else stage2_frozen).append(n)
    print(f"[ParamCheck] stage2_trainable_count: {len(stage2_trainable)}")
    print(f"[ParamCheck] stage2_frozen_count: {len(stage2_frozen)}")
    print(f"[ParamCheck] sample_trainable: {stage2_trainable[:10]}")
    print(f"[ParamCheck] sample_frozen: {stage2_frozen[:10]}")
    if len(stage2_trainable) == 0:
        raise RuntimeError("Stage2 param check failed: no trainable stage2 params.")

    with open(os.path.join(args.out, "stage2_trainable_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"trainable={trainable}\n")
        f.write(f"total={total}\n")
        f.write(f"stage1_trainable={stage1_trainable}\n")
        f.write(f"non_lora_trainable={non_lora_trainable}\n")

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # One-batch label validity check (no Trainer instantiation)
    dl = DataLoader(
        ds,
        batch_size=args.bsz,
        shuffle=False,
        collate_fn=data_collator,
    )
    batch = next(iter(dl))
    labels = batch.get("labels")
    if labels is None:
        raise RuntimeError("Label check failed: batch has no labels.")
    print(f"[LabelCheck] dtype={labels.dtype}")
    print(f"[LabelCheck] min={labels.min().item()} max={labels.max().item()}")
    valid = (labels != -100)
    count_valid = int(valid.sum().item())
    count_total = labels.numel()
    unique_labels = torch.unique(labels[valid])[:20] if count_valid > 0 else torch.tensor([])
    print(f"[LabelCheck] count_valid={count_valid} count_total={count_total}")
    print(f"[LabelCheck] unique_labels_sample={unique_labels.tolist()}")
    vocab_size = tok.vocab_size
    if labels.dtype != torch.int64:
        raise RuntimeError("Label check failed: labels dtype is not int64.")
    if labels.min().item() < -100 or labels.max().item() >= vocab_size:
        raise RuntimeError("Label check failed: labels out of range.")
    if count_valid == 0:
        raise RuntimeError("Label check failed: all labels are masked (-100).")

    # Move model and batch to GPU for fast checks (avoids slow CPU forward/backward)
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    model = model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Loss reduction + stage2 grad check
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    with torch.no_grad():
        model.set_adapter("stage1")
        out1 = model(**batch).logits
        try:
            model.set_adapter(["stage1", "stage2"])
        except Exception:
            model.set_adapter("stage2")
        out2 = model(**batch).logits
    diff_norm = (out2 - out1).float().norm().item()
    print(f"[AdapterState] logit_diff_norm: {diff_norm}")
    if diff_norm == 0.0:
        raise RuntimeError("Stage2 appears inactive: logit_diff_norm == 0.")
    outputs = model(**batch)
    loss = outputs.loss
    logits = outputs.logits
    seq_len = int(batch["input_ids"].shape[-1])
    non_masked = int((labels != -100).sum().item())
    print(f"[LossCheck] loss={loss.item()} seq_len={seq_len} non_masked={non_masked}")
    print(f"[LossCheck] loss_finite={torch.isfinite(loss).item()}")
    print(f"[LossCheck] logits_finite={torch.isfinite(logits).all().item()}")

    # Manual backward + grad stats
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out = model(**batch)
    loss_manual = out.loss
    print(f"[ManualBackward] loss: {loss_manual.item()}")
    loss_manual.backward()
    def stage2_grad_summary(m):
        total_sq = 0.0
        nz = 0
        cnt = 0
        for n, p in m.named_parameters():
            if p.requires_grad and "stage2" in n and p.grad is not None:
                g = p.grad.detach()
                gn = g.norm().item()
                cnt += 1
                if gn > 0:
                    nz += 1
                total_sq += (gn * gn)
        total = total_sq ** 0.5
        print(f"[GradCheck] stage2_grad_norm_l2={total:.6f} nonzero_tensors={nz}/{cnt}")

    stage2_grad_summary(model)
    found = False
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora" in n.lower() or "stage2" in n.lower()):
            g = p.grad
            print(
                "[ManualBackward] first_trainable:",
                n,
                "grad_is_none:",
                (g is None),
                "grad_norm:",
                (g.norm().item() if g is not None else None),
                "grad_finite:",
                (torch.isfinite(g).all().item() if g is not None else None),
            )
            found = True
            break
    print(f"[ManualBackward] found_trainable_param: {found}")

    # Smoke test: overfit 8 examples
    if args.smoke_test:
        smoke_lr = args.lr if args.smoke_lr is None else args.smoke_lr
        smoke_ds = ds.select(range(min(8, len(ds))))
        smoke_args = TrainingArguments(
            output_dir=os.path.join(args.out, "smoke_test"),
            per_device_train_batch_size=args.bsz,
            gradient_accumulation_steps=args.ga,
            learning_rate=smoke_lr,
            max_steps=args.smoke_steps,
            warmup_ratio=0.0,
            lr_scheduler_type="cosine",
            logging_steps=max(1, args.smoke_steps // 5),
            save_strategy="no",
            report_to=None,
            fp16=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=args.seed,
        )
        smoke_trainer = Trainer(
            model=model,
            args=smoke_args,
            train_dataset=smoke_ds,
            data_collator=data_collator,
        )
        smoke_trainer.train()
        losses = [x["loss"] for x in smoke_trainer.state.log_history if "loss" in x]
        if len(losses) >= 2:
            start_loss = losses[0]
            end_loss = losses[-1]
            if end_loss > 0.9 * start_loss:
                raise RuntimeError(
                    "Smoke test failed: loss did not drop by >=10%. "
                    "Check tokenization, masking, and adapter activation."
                )
        else:
            raise RuntimeError("Smoke test failed: insufficient loss logs.")
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="no",
        report_to=None,
        fp16=bool(cuda_ok and args.fp16),
        bf16=bool(cuda_ok and args.bf16),
        max_grad_norm=args.max_grad_norm,
        adam_epsilon=1e-6,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    class LrPrintCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                opt = kwargs.get("optimizer")
                if opt and opt.param_groups:
                    print(f"[LRStep] optimizer_lr0={opt.param_groups[0]['lr']}")

    trainer = DebugTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=data_collator,
        callbacks=[LrPrintCallback()],
    )
    trainer.create_optimizer()
    print(f"[OptLR after create_optimizer] {trainer.optimizer.param_groups[0]['lr']}")
    trainer.create_scheduler(num_training_steps=targs.max_steps)
    print(f"[Sch last lr] {trainer.lr_scheduler.get_last_lr()[0]}")
    print(f"[OptLR after create_scheduler] {trainer.optimizer.param_groups[0]['lr']}")
    print(f"[max_steps] {targs.max_steps}")

    # First-batch sanity from Trainer dataloader
    first_batch = next(iter(trainer.get_train_dataloader()))
    fb_labels = first_batch.get("labels")
    if fb_labels is not None:
        fb_valid = (fb_labels != -100)
        print(f"[BatchCheck] input_ids.shape={first_batch['input_ids'].shape}")
        print(f"[BatchCheck] labels.min={fb_labels.min().item()} max={fb_labels.max().item()}")
        print(f"[BatchCheck] num_valid={int(fb_valid.sum().item())} total={fb_labels.numel()}")
    trainer.train()

    # Save Stage-2 adapter only
    model.save_pretrained(args.out, adapter_name="stage2")

    # Generation regression check
    if args.eval_prompts_json:
        with open(args.eval_prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("--eval_prompts_json must be a non-empty JSON list.")

        device = next(model.parameters()).device
        inputs = build_generation_inputs(tok, prompts, device)

        def generate_with_adapter(adapter_name):
            if adapter_name == "base":
                with model.disable_adapter():
                    return model.generate(**inputs, max_new_tokens=64)
            model.set_adapter(adapter_name)
            return model.generate(**inputs, max_new_tokens=64)

        base_out = generate_with_adapter("base")
        model.set_adapter("stage1")
        stage1_out = model.generate(**inputs, max_new_tokens=64)
        try:
            model.set_adapter(["stage1", "stage2"])
        except Exception as exc:
            raise RuntimeError("PEFT does not support stacked adapters in this version.") from exc
        stage2_out = model.generate(**inputs, max_new_tokens=64)

        def decode(outputs):
            return [tok.decode(o, skip_special_tokens=True) for o in outputs]

        base_text = decode(base_out)
        stage1_text = decode(stage1_out)
        stage2_text = decode(stage2_out)

        differs = any(a != b for a, b in zip(stage1_text, stage2_text))
        if not differs:
            raise RuntimeError("Stage2 regression check failed: stage1 == stage2 outputs.")

        gen_report = {
            "prompts": prompts,
            "base": base_text,
            "stage1": stage1_text,
            "stage2": stage2_text,
        }
        with open(os.path.join(args.out, "stage2_generations.json"), "w", encoding="utf-8") as f:
            json.dump(gen_report, f, indent=2)

    manifest = {
        "base_model": args.model,
        "adapters": [
            {"name": "stage1", "path": os.path.abspath(args.stage1_adapter)},
            {"name": "stage2", "path": os.path.abspath(args.out)},
        ],
        "adapter_order": ["stage1", "stage2"],
        "notes": "M2 = base + stage1 + stage2",
    }
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Saved Stage-2 adapter to {args.out}")
    print(f"✅ Wrote manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
