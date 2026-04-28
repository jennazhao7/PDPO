#!/usr/bin/env python3
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def resolve_adapter_path(path: str) -> str:
    """Accept either adapter root or a run directory containing stage2/adapter_config.json."""
    cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(cfg):
        return path
    stage2 = os.path.join(path, "stage2")
    cfg2 = os.path.join(stage2, "adapter_config.json")
    if os.path.exists(cfg2):
        return stage2
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--stage1_adapter", required=True)
    ap.add_argument("--stage2_adapter", required=True)
    ap.add_argument("--stage1_name", default="stage1")
    ap.add_argument("--stage2_name", default="stage2")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )

    stage1_path = resolve_adapter_path(args.stage1_adapter)
    stage2_path = resolve_adapter_path(args.stage2_adapter)
    print(f"Resolved stage1 adapter: {stage1_path}")
    print(f"Resolved stage2 adapter: {stage2_path}")

    model = PeftModel.from_pretrained(base, stage1_path, adapter_name=args.stage1_name)
    model.load_adapter(stage2_path, adapter_name=args.stage2_name)

    norms = []
    total_params = 0
    for name, p in model.named_parameters():
        if args.stage2_name in name and "lora" in name:
            n = float(p.detach().float().norm().item())
            norms.append((name, n, p.numel()))
            total_params += p.numel()

    norms.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(norms)} stage2 LoRA tensors, total params={total_params}")
    if not norms:
        return

    vals = torch.tensor([x[1] for x in norms], dtype=torch.float32)
    near_zero = int((vals < 1e-6).sum().item())

    print(
        "norm stats: "
        f"min={vals.min().item():.6g} "
        f"max={vals.max().item():.6g} "
        f"mean={vals.mean().item():.6g} "
        f"median={vals.median().item():.6g}"
    )
    print(f"near-zero tensors (<1e-6): {near_zero}/{len(norms)}")

    print("\nTop 20 tensor norms:")
    for name, n, sz in norms[:20]:
        print(f"{name:120s} norm={n:.6g} numel={sz}")


if __name__ == "__main__":
    main()

