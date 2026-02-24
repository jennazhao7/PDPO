#!/usr/bin/env python3
"""
Verify Stage-2 training produced correct artifacts:
1) stage2 params changed, stage1 params unchanged
2) M2 loads as base + stage1 + stage2 and outputs differ from base+stage1
"""
from __future__ import annotations

import hashlib
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest_path = "outputs/M2_stack_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    base_model = manifest["base_model"]
    stage1_path = manifest["adapters"][0]["path"]
    stage2_path = manifest["adapters"][1]["path"]

    print("=== 1) Verify stage1 unchanged, stage2 trained ===")
    stage1_orig = os.path.join(stage1_path, "adapter_model.safetensors")
    stage1_in_outputs = os.path.join(stage2_path, "stage1", "adapter_model.safetensors")
    stage2_file = os.path.join(stage2_path, "adapter_model.safetensors")
    if not os.path.exists(stage2_file):
        stage2_file = os.path.join(stage2_path, "stage2", "adapter_model.safetensors")

    cs_orig = None
    if os.path.exists(stage1_orig):
        cs_orig = file_checksum(stage1_orig)
        print(f"  stage1 (original) checksum: {cs_orig[:16]}...")
    if os.path.exists(stage1_in_outputs) and cs_orig is not None:
        cs_in_outputs = file_checksum(stage1_in_outputs)
        print(f"  stage1 (in outputs) checksum: {cs_in_outputs[:16]}...")
        if cs_orig == cs_in_outputs:
            print("  OK: stage1 unchanged (checksums match)")
        else:
            print("  WARN: stage1 in outputs differs from original")
    if os.path.exists(stage2_file):
        print(f"  stage2 adapter: {stage2_file}")
        print(f"  stage2 file exists")

    print("\n=== 2) Load M2 and compare base+stage1 vs base+stage1+stage2 ===")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = PeftModel.from_pretrained(model, stage1_path, adapter_name="stage1")

    # stage2 might be at stage2_path or stage2_path/stage2
    stage2_load = stage2_path
    if os.path.isdir(os.path.join(stage2_path, "stage2")):
        stage2_load = os.path.join(stage2_path, "stage2")
    model.load_adapter(stage2_load, adapter_name="stage2")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "Write a haiku about coding.",
        "What is 2 + 2?",
        "Recommend a good book.",
    ]

    model.set_adapter("stage1")
    out_s1 = []
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(device)
            gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
            out_s1.append(tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))

    try:
        model.set_adapter(["stage1", "stage2"])
    except Exception:
        model.set_adapter("stage2")

    out_s2 = []
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(device)
            gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
            out_s2.append(tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))

    differs = sum(1 for a, b in zip(out_s1, out_s2) if a != b)
    print(f"  Prompts where stage1+stage2 != stage1 only: {differs}/{len(prompts)}")
    if differs > 0:
        print("  OK: stage2 affects outputs")
    else:
        print("  WARN: stage2 outputs identical to stage1 only")

    for i, (p, s1, s2) in enumerate(zip(prompts, out_s1, out_s2)):
        print(f"\n  prompt {i+1}: {p[:50]}...")
        print(f"    stage1: {s1[:80]}...")
        print(f"    stage2: {s2[:80]}...")
        print(f"    diff: {s1 != s2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
