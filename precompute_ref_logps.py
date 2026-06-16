#!/usr/bin/env python3
import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from experiments import aggregate


def slug_model(model: str) -> str:
    return model.replace("/", "--").replace(":", "_")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_template(template: str, cell: Dict[str, Any]) -> str:
    values = dict(cell)
    eps = float(cell["eps"])
    values["eps_slug"] = str(eps).rstrip("0").rstrip(".")
    return template.format(**values)


def unique_dataset_model_jobs(manifest: Dict[str, Any], train_template: str) -> List[Tuple[str, str, Path]]:
    jobs: Dict[Tuple[str, str, str], Tuple[str, str, Path]] = {}
    for cell in aggregate.expand_manifest(manifest):
        dataset = str(cell["dataset"])
        base_model = str(cell["base_model"])
        train_path = Path(format_template(train_template, cell))
        key = (dataset, base_model, str(train_path))
        jobs[key] = (dataset, base_model, train_path)
    return sorted(jobs.values(), key=lambda item: item[0])


@torch.no_grad()
def response_logprob(model, tok, prompt: str, response: str, device: torch.device, max_len: int) -> float:
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tok(response, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + response_ids)[-max_len:]
    if len(ids) < 2:
        return 0.0
    prompt_len = min(len(prompt_ids), len(ids) - 1)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    start = max(prompt_len - 1, 0)
    return float(tok_lp[start:].sum().item())


def upload_to_bucket(local_path: Path, bucket: str, remote_name: str, dry_run: bool) -> str:
    remote = f"{bucket.rstrip('/')}/ref_logps/{remote_name}"
    if dry_run:
        print(f"[dry-run] gcloud storage cp {local_path} {remote}")
        return remote
    subprocess.run(["gcloud", "storage", "cp", str(local_path), remote], check=True)
    return remote


def compute_cache(
    dataset: str,
    base_model: str,
    train_path: Path,
    out_dir: Path,
    max_len: int,
    force: bool,
) -> Path:
    cache_path = out_dir / f"{dataset}_{slug_model(base_model)}_ref_logps.jsonl"
    if cache_path.exists() and not force:
        print(f"[skip] cache exists: {cache_path}")
        return cache_path

    rows = read_jsonl(train_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_fast = "open_llama" not in base_model.lower()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device).eval()

    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            prompt = str(row["prompt"])
            chosen = str(row["chosen"])
            rejected = str(row["rejected"])
            record = {
                "dataset": dataset,
                "base_model": base_model,
                "source": str(train_path),
                "idx": idx,
                "prompt_hash": prompt_hash(prompt),
                "chosen_ref_logp": response_logprob(model, tok, prompt, chosen, device, max_len),
                "rejected_ref_logp": response_logprob(model, tok, prompt, rejected, device, max_len),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(cache_path)
    print(f"[ok] wrote {cache_path}")
    return cache_path


def parse_args() -> argparse.Namespace:
    root = aggregate.repo_root()
    ap = argparse.ArgumentParser(description="Precompute base/reference logprobs once and upload to GCS.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--train-template", default="stage2_debugging/preprocessing/d2_rr_flipped_{dataset}_eps{eps}_seed{seed}.jsonl")
    ap.add_argument("--out-dir", default=str(root / "experiments/ref_logps"))
    ap.add_argument("--bucket", default=config.BUCKET)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config.validate_runtime_config(require_bucket=True)
    manifest = aggregate.load_manifest(Path(args.manifest))
    uploaded = []
    for dataset, base_model, train_path in unique_dataset_model_jobs(manifest, args.train_template):
        if not train_path.exists():
            print(f"[missing] {train_path}; skipping cache for {dataset}/{base_model}")
            continue
        cache = compute_cache(dataset, base_model, train_path, Path(args.out_dir), args.max_len, args.force)
        remote = upload_to_bucket(cache, args.bucket, cache.name, args.dry_run)
        uploaded.append({"local": str(cache), "remote": remote})
    manifest_path = Path(args.out_dir) / "ref_logps_manifest.json"
    manifest_path.write_text(json.dumps({"uploaded": uploaded}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
