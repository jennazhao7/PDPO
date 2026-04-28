#!/usr/bin/env python3
"""
generate_responses.py
=====================
Load a base model once, hot-swap LoRA adapters, and generate responses for
held-out prompts.  All adapters share the same decoding seed per prompt so
that differences reflect adapter behaviour, not sampling noise.

Adapter spec: --adapters name1=path1,name2=path2,...
  - path may be a directory with adapter_config.json  (direct adapter)
  - path may be an M2_manifest.json file              (reads first adapter)
  - path may be a directory containing fresh_lora/    (map_retrain layout)

Output: one JSONL per adapter under --output_dir/<adapter_name>_responses.jsonl
  {prompt_id, prompt, response, adapter_name, decoding_seed}

Usage:
  python generate_responses.py \
      --base_model Qwen/Qwen2.5-3B \
      --adapters sb_fresh=/path/to/sb,map_retrain=/path/to/map,mle_fresh=/path/to/mle \
      --prompts data/pku_test_200prompts.jsonl \
      --output_dir responses/pku_eps0.5_seed42/ \
      --max_new_tokens 256 \
      [--n 50]   # smoke-test subset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ---------------------------------------------------------------------------
# Adapter path resolution
# ---------------------------------------------------------------------------

def resolve_adapter_path(raw_path: str, _visited: frozenset | None = None) -> str:
    """
    Given a user-supplied path, return the directory containing adapter_config.json.
    Supports:
      1. Direct path to adapter dir (has adapter_config.json)
      2. Directory with stage2/     → sb_fresh layout (checked BEFORE manifest to avoid cycles)
      3. Directory with fresh_lora/ → map_retrain / mle_fresh layout
      4. M2_manifest.json file      → reads first adapter path (cycle-safe)
      5. Directory with M2_manifest.json → reads first adapter path (cycle-safe)
    """
    if _visited is None:
        _visited = frozenset()

    p = Path(raw_path).resolve()
    p_str = str(p)

    if p_str in _visited:
        # Self-referential manifest — fall through to sub-directory checks below
        pass
    else:
        _visited = _visited | {p_str}

        # Case: it's a manifest JSON file
        if p.is_file() and p.suffix == ".json":
            manifest = json.loads(p.read_text())
            adapter_path = manifest["adapters"][0]["path"]
            return resolve_adapter_path(adapter_path, _visited)

    # Case: direct adapter dir
    if (p / "adapter_config.json").exists():
        return str(p)

    # Case: stage2/ subdirectory  — check BEFORE manifest to avoid sb_fresh self-referential loop
    if (p / "stage2" / "adapter_config.json").exists():
        return str(p / "stage2")

    # Case: fresh_lora/ subdirectory
    if (p / "fresh_lora" / "adapter_config.json").exists():
        return str(p / "fresh_lora")

    # Case: directory with M2_manifest.json — only follow if not already visited
    manifest_f = p / "M2_manifest.json"
    if manifest_f.exists() and str(manifest_f.resolve()) not in _visited:
        manifest = json.loads(manifest_f.read_text())
        adapter_path = manifest["adapters"][0]["path"]
        return resolve_adapter_path(adapter_path, _visited | {str(manifest_f.resolve())})

    # Last resort: return as-is
    return str(p)


def parse_adapter_spec(spec: str) -> dict[str, str]:
    """Parse 'name1=path1,name2=path2' → {name: resolved_path}."""
    adapters = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"Adapter spec must be name=path, got: {part!r}")
        name, path = part.split("=", 1)
        name = name.strip()
        path = path.strip()
        resolved = resolve_adapter_path(path)
        print(f"[Adapter] {name!r} → {resolved}", flush=True)
        adapters[name] = resolved
    return adapters


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model_with_adapters(base_model_id: str, adapter_map: dict[str, str], device: torch.device):
    """
    Load the base model once, then attach all adapters as named PEFT adapters.
    Returns (model, tokenizer).
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[Load] base model {base_model_id} on {device} (dtype={dtype})", flush=True)

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map={"": device} if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(device)

    # Attach adapters
    first = True
    for adapter_name, adapter_path in adapter_map.items():
        print(f"[Load] attaching adapter {adapter_name!r} from {adapter_path}", flush=True)
        if first:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
            first = False
        else:
            # peft ≥0.12 validates load_adapter() paths against HF repo-id format,
            # rejecting absolute local paths. local_files_only=True bypasses this.
            model.load_adapter(adapter_path, adapter_name=adapter_name,
                               local_files_only=True)

    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(
    model,
    tok,
    adapter_name: str,
    prompt: str,
    decoding_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    """Generate a single response with the specified adapter active."""
    model.set_adapter(adapter_name)

    # Encode prompt only
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # Fix seed for this (prompt, adapter) pair — adapter doesn't matter here
    # because we want the same stochastic path across adapters for a given prompt
    set_seed(decoding_seed)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )

    # Decode only the new tokens
    new_tokens = out[0, prompt_len:]
    response = tok.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(path: str, n: int | None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if n is not None:
        rows = rows[:n]
    # Ensure every row has a prompt_id
    for i, r in enumerate(rows):
        if "prompt_id" not in r:
            r["prompt_id"] = r.get("id", f"row_{i:05d}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate responses from multiple LoRA adapters.")
    ap.add_argument("--base_model", required=True,
                    help="HF model ID or local path for the base model.")
    ap.add_argument("--adapters", required=True,
                    help="Comma-separated name=path specs, e.g. sb_fresh=...,map_retrain=...")
    ap.add_argument("--prompts", required=True,
                    help="Path to JSONL file with held-out test prompts.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write per-adapter JSONL files.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="Added to prompt_id hash to form decoding seed. Change to get different samples.")
    ap.add_argument("--n", type=int, default=None,
                    help="Smoke-test: only use first N prompts.")
    ap.add_argument("--device", type=str, default=None,
                    help="Device override, e.g. 'cuda:0'. Default: auto.")
    args = ap.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Config] device={device} max_new_tokens={args.max_new_tokens}", flush=True)

    adapter_map = parse_adapter_spec(args.adapters)
    prompts = load_prompts(args.prompts, args.n)
    print(f"[Config] {len(prompts)} prompts, {len(adapter_map)} adapters", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model + all adapters once
    model, tok = load_base_model_with_adapters(args.base_model, adapter_map, device)

    # Open output files
    out_files = {}
    for adapter_name in adapter_map:
        out_path = os.path.join(args.output_dir, f"{adapter_name}_responses.jsonl")
        out_files[adapter_name] = open(out_path, "w", encoding="utf-8")
        print(f"[Output] {adapter_name} → {out_path}", flush=True)

    try:
        for pi, prompt_row in enumerate(prompts):
            prompt_id = prompt_row["prompt_id"]
            prompt_text = prompt_row["prompt"]

            # Deterministic seed per prompt, shared across all adapters
            decoding_seed = (hash(str(prompt_id)) & 0x7FFFFFFF) + args.seed_offset

            for adapter_name in adapter_map:
                response = generate_response(
                    model, tok, adapter_name,
                    prompt=prompt_text,
                    decoding_seed=decoding_seed,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "response": response,
                    "adapter_name": adapter_name,
                    "decoding_seed": decoding_seed,
                }
                out_files[adapter_name].write(json.dumps(record, ensure_ascii=False) + "\n")
                out_files[adapter_name].flush()

            if (pi + 1) % 10 == 0 or (pi + 1) == len(prompts):
                print(f"  [{pi+1}/{len(prompts)}] done", flush=True)
    finally:
        for f in out_files.values():
            f.close()

    print(f"\n[Done] responses written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
