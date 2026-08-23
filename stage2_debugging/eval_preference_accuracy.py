#!/usr/bin/env python3
"""
Compute DPO implicit reward accuracy and ECE on test_pref.jsonl.

Metric: reward(y) = log π(y|x) - log π_ref(y|x)
  where π     = base + stage1 + stage2  (the trained policy)
  and   π_ref = base + stage1           (stage1 = frozen reference for stage2)

This is naturally length-normalized (reference sees same length pressure)
and matches exactly what the DPO training loss optimizes.

accuracy = Σ [reward(chosen) > reward(rejected)] / N
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from matched_core import load_reference_cache
from metric_primitives import canonical_json_hash, preference_accuracy, preference_pair_key
from ref_logprob_core import (
    LOGPROB_DTYPE,
    REFERENCE_DTYPE,
    TRUNCATION_MODE,
    encode_prompt_response,
    torch_batched_response_logprobs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from heldout_guard import assert_heldout_verified  # noqa: E402


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def _resolve_adapter_path(path, manifest_dir):
    import os
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path


def load_model(manifest_path, device):
    import os
    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    base = pick(m, ["base_model", "base_model_id", "base_model_name_or_path"])
    adapters = m.get("adapters", [])
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    if len(adapters) == 1:
        s1 = None
        s2 = _resolve_adapter_path(adapters[0]["path"], manifest_dir)
    else:
        s1 = _resolve_adapter_path(adapters[0]["path"], manifest_dir)
        s2 = _resolve_adapter_path(adapters[1]["path"], manifest_dir)

    tok = AutoTokenizer.from_pretrained(
        base, use_fast="open_llama" not in str(base).lower()
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_name = os.environ.get("PDPO_PRECISION", "fp32")
    if dtype_name != "fp32":
        raise ValueError(f"Unsupported locked evaluation precision: {dtype_name}")
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    
    if s1:
        model = PeftModel.from_pretrained(model, s1, adapter_name="stage1")
        model.load_adapter(s2, adapter_name="stage2")
    else:
        # If there's only one adapter (e.g. map_retrain), just call it stage2
        model = PeftModel.from_pretrained(model, s2, adapter_name="stage2")
        
    model = model.to(device).eval()
    return model, tok, (s1 is not None)


@torch.no_grad()
def resp_logprob_sum(model, active_adapters, tok, prompt, resp, device, max_len=512):
    """Sum of token log-probs over response tokens only."""
    if active_adapters:
        try:
            model.set_adapter(active_adapters)
        except Exception:
            model.set_adapter(active_adapters[-1])
    ids, prompt_len = encode_prompt_response(tok, prompt, resp, max_len)
    if len(ids) < 2:
        return 0.0
    values = torch_batched_response_logprobs(
        model, [ids], [prompt_len], tok.pad_token_id, device
    )
    return float(values[0].item())


def policy_and_reference_logprob(
    model, tok, prompt, resp, device, max_len, has_stage1: bool, cached_ref=None
):
    """Policy from a forward pass; reference from `cached_ref` when supplied.

    The base model is identical across every checkpoint, epoch, and arm, and this script always
    scores the TRUE unflipped orientation, so the reference pass recomputes one fixed value per
    (prompt, response) 60 times over. Serving it from the hash-validated cache removes exactly
    half the forward passes here.
    """
    if has_stage1:
        # policy = stage1 + stage2
        log_policy = resp_logprob_sum(model, ["stage1", "stage2"], tok, prompt, resp, device, max_len)
        if cached_ref is not None:
            return log_policy, cached_ref
        # reference = stage1 only
        log_ref = resp_logprob_sum(model, "stage1", tok, prompt, resp, device, max_len)
    else:
        # policy = single adapter (we named it "stage2" above)
        log_policy = resp_logprob_sum(model, ["stage2"], tok, prompt, resp, device, max_len)
        if cached_ref is not None:
            return log_policy, cached_ref
        # reference = base model (disable adapters)
        with model.disable_adapter():
            log_ref = resp_logprob_sum(model, [], tok, prompt, resp, device, max_len)
            
    return log_policy, log_ref


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_ece(conf, correct, n_bins=10):
    ece = 0.0
    bins = []
    n = len(conf)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            bins.append({"bin": b, "count": 0, "acc": None, "conf": None})
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        cavg = sum(conf[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - cavg)
        bins.append({"bin": b, "count": len(idx), "acc": acc, "conf": cavg})
    return ece, bins


def prompt_hashes(rows):
    return {
        hashlib.sha256(str(row["prompt"]).strip().encode("utf-8")).hexdigest()
        for row in rows
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_artifacts_sha256(manifest_path):
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    manifest_dir = Path(manifest_path).resolve().parent
    files = []
    for adapter in manifest.get("adapters", []):
        adapter_path = Path(adapter["path"])
        if not adapter_path.is_absolute():
            adapter_path = manifest_dir / adapter_path
        adapter_path = Path(_resolve_adapter_path(str(adapter_path), str(manifest_dir)))
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            candidate = adapter_path / name
            if candidate.is_file():
                files.append(candidate)
    if not files:
        raise FileNotFoundError("No adapter artifacts found for provenance hashing")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: str(value)):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def score_rows(model, tok, rows, device, max_len, has_stage1, progress=False, cached=None):
    """`cached` is (chosen_sums, rejected_sums) aligned index-for-index with `rows`."""
    calibrated, raw, pair_keys = [], [], []
    for i, row in enumerate(rows):
        c_ref = cached[0][i] if cached else None
        r_ref = cached[1][i] if cached else None
        policy_chosen, ref_chosen = policy_and_reference_logprob(
            model, tok, row["prompt"], row["chosen"], device, max_len, has_stage1, c_ref
        )
        policy_rejected, ref_rejected = policy_and_reference_logprob(
            model, tok, row["prompt"], row["rejected"], device, max_len, has_stage1, r_ref
        )
        raw_gap = policy_chosen - policy_rejected
        raw.append(raw_gap)
        calibrated.append(raw_gap - (ref_chosen - ref_rejected))
        pair_keys.append(preference_pair_key(row))
        if progress and (i + 1) % 100 == 0:
            print(
                f"  [{i+1}/{len(rows)}] running_acc="
                f"{preference_accuracy(calibrated):.3f}",
                flush=True,
            )
    if not calibrated:
        raise ValueError("preference evaluation requires at least one row")
    return {
        "n": len(rows),
        "pair_keys_sha256": canonical_json_hash(pair_keys),
        "reference_calibrated_dpo_margin": {
            "accuracy": preference_accuracy(calibrated),
            "mean_gap": sum(calibrated) / len(calibrated),
            "gaps": calibrated,
        },
        "raw_policy_preference_gap": {
            "accuracy": preference_accuracy(raw),
            "mean_gap": sum(raw) / len(raw),
            "gaps": raw,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train_jsonl")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument(
        "--ref_logps",
        default=None,
        help="Eval reference cache from experiments/build_eval_ref_cache.py. When given, the "
             "base-model reference forward passes are served from it instead of recomputed.",
    )
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument(
        "--heldout-verified-against",
        action="append",
        default=[],
        help="Additional training partition(s) the test set must be disjoint from.",
    )
    ap.add_argument(
        "--i-know-this-is-not-heldout",
        action="store_true",
        help="Diagnostic only: accept an unverified test set. The result is marked invalid.",
    )
    args = ap.parse_args()

    # Fail closed: a test set that cannot be shown disjoint from this run's training data is
    # treated as unsafe, not as safe. --train_jsonl is always supplied by run_queue.eval_command.
    training_paths = [Path(p) for p in ([args.train_jsonl] if args.train_jsonl else [])]
    training_paths += [Path(p) for p in args.heldout_verified_against]
    heldout_verification = assert_heldout_verified(
        Path(args.test_jsonl),
        training_paths,
        role="test",
        allow_unverified=args.i_know_this_is_not_heldout,
    )

    rows = [json.loads(x) for x in open(args.test_jsonl, "r", encoding="utf-8") if x.strip()]
    train_rows = (
        [json.loads(x) for x in open(args.train_jsonl, "r", encoding="utf-8") if x.strip()]
        if args.train_jsonl
        else None
    )
    device = torch.device("cuda")
    model, tok, has_stage1 = load_model(args.manifest, device)

    # Cache loaded per split so the returned lists align index-for-index with that split's rows.
    # Every row's prompt/chosen/rejected hashes are validated, so a misalignment is a hard error.
    heldout_cached = train_cached = None
    reference_cache_stats = {"used": False}
    if args.ref_logps:
        _man = json.load(open(args.manifest, "r", encoding="utf-8"))
        _common = dict(
            expected_base_model=_man["base_model"],
            expected_scoring_dtype=REFERENCE_DTYPE,
            expected_truncation=TRUNCATION_MODE,
            expected_max_len=args.max_len,
            expected_rr_eps=None,
            expected_cache_role="eval_reference_true_orientation",
        )
        heldout_cached = load_reference_cache(args.ref_logps, rows, **_common)
        _skipped = 2 * len(rows)
        if train_rows is not None:
            train_cached = load_reference_cache(args.ref_logps, train_rows, **_common)
            _skipped += 2 * len(train_rows)
        reference_cache_stats = {
            "used": True, "path": str(args.ref_logps), "heldout_hits": len(rows),
            "train_hits": len(train_rows) if train_rows is not None else 0,
            "hit_rate": 1.0, "reference_forwards_skipped": _skipped,
        }
        print(f"Reference cache: {_skipped} reference forwards skipped", flush=True)

    heldout = score_rows(
        model, tok, rows, device, args.max_len, has_stage1, progress=True,
        cached=heldout_cached
    )
    train = (
        score_rows(model, tok, train_rows, device, args.max_len, has_stage1,
                   cached=train_cached)
        if train_rows is not None
        else None
    )
    margins = heldout["reference_calibrated_dpo_margin"]["gaps"]
    confs = [sigmoid(margin) for margin in margins]
    corr = [1 if margin > 0 else 0 for margin in margins]

    acc = preference_accuracy(margins)
    ece, bins = compute_ece(confs, corr, args.n_bins)
    positive_margins = sum(1 for margin in margins if margin > 0)
    negative_margins = sum(1 for margin in margins if margin < 0)
    zero_margins = len(margins) - positive_margins - negative_margins

    out = {
        "manifest": args.manifest,
        "metric": "dpo_implicit_reward",
        "n": len(rows),
        "accuracy": acc,
        "acc": acc,
        "train_accuracy": (
            train["reference_calibrated_dpo_margin"]["accuracy"] if train else None
        ),
        "train_gap": (
            train["reference_calibrated_dpo_margin"]["mean_gap"] if train else None
        ),
        "heldout_accuracy": acc,
        "heldout_gap": heldout["reference_calibrated_dpo_margin"]["mean_gap"],
        "heldout_verification": heldout_verification,
        "reference_cache": reference_cache_stats,
        "splits": {"train": train, "heldout": heldout},
        "provenance": {
            "manifest_sha256": file_sha256(args.manifest),
            "adapter_artifacts_sha256": adapter_artifacts_sha256(args.manifest),
            "train_jsonl_sha256": (
                file_sha256(args.train_jsonl)
                if args.train_jsonl
                else None
            ),
            "test_jsonl_sha256": file_sha256(args.test_jsonl),
            "scoring_config_sha256": canonical_json_hash(
                {
                    "max_len": args.max_len,
                    "precision": "float32",
                    "truncation": TRUNCATION_MODE,
                    "attacks": [
                        "reference_calibrated_dpo_margin",
                        "raw_policy_preference_gap",
                    ],
                }
            ),
        },
        "ece": ece,
        "mean_margin": sum(margins) / len(margins),
        "margins": margins,
        "margin_sign_distribution": {
            "positive": positive_margins,
            "negative": negative_margins,
            "zero": zero_margins,
        },
        "precision": {
            "model_weights": "float32",
            "logprobs": LOGPROB_DTYPE,
            "reward_difference": LOGPROB_DTYPE,
        },
        "truncation": TRUNCATION_MODE,
        "max_len": args.max_len,
        "bins": bins,
    }
    output = Path(args.out_json)
    if "outputs" in output.resolve().parts:
        raise ValueError("refusing to write evaluation artifact under outputs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"accuracy": acc, "ece": ece, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
