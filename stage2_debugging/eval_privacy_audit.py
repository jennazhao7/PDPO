#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from matched_core import load_reference_cache_detail
from metric_primitives import (
    MIA_STATISTICS,
    canonical_json_hash,
    deterministic_sample,
    preference_accuracy,
    preference_mia_statistics,
    preference_pair_key,
    privacy_metrics,
)
from ref_logprob_core import (
    LOGPROB_DTYPE,
    REFERENCE_DTYPE,
    TRUNCATION_MODE,
    encode_prompt_response,
    torch_batched_response_token_logprobs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from heldout_guard import assert_heldout_verified  # noqa: E402

def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None

def _resolve_adapter_path(path, manifest_dir):
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path

def load_model(manifest_path, device):
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
        model = PeftModel.from_pretrained(model, s2, adapter_name="stage2")
        
    model = model.to(device).eval()
    return model, tok, (s1 is not None)

@torch.no_grad()
def resp_logprob_detail(model, active_adapters, tok, prompt, resp, device, max_len=512):
    """Return the response-only logprob sum and the per-token logprobs behind it.

    The sum is computed with the same masked-multiply reduction as
    torch_batched_response_logprobs, so it is bit-identical to the previous scalar path.
    """
    if active_adapters:
        try:
            model.set_adapter(active_adapters)
        except Exception:
            model.set_adapter(active_adapters[-1])
    ids, prompt_len = encode_prompt_response(tok, prompt, resp, max_len)
    if len(ids) < 2:
        return 0.0, []
    values, mask = torch_batched_response_token_logprobs(
        model, [ids], [prompt_len], tok.pad_token_id, device
    )
    total = float((values * mask).sum(dim=1)[0].item())
    per_token = [float(value) for value in values[0][mask[0].bool()].tolist()]
    return total, per_token

def policy_and_reference_logprob(
    model, tok, prompt, resp, device, max_len, has_stage1: bool, cached_ref=None
):
    """Policy logprobs always come from a forward pass. The reference comes from `cached_ref`
    when supplied.

    The base model is identical across every checkpoint, epoch, and arm, and eval always scores
    the TRUE unflipped orientation, so the reference pass recomputes the same value 60 times.
    When a cache is supplied the reference forward is skipped -- exactly half the forward passes
    in this script. `cached_ref` is (sum, per_token) produced by the same
    torch_batched_response_token_logprobs path, so substituting it is numerically inert.
    """
    if has_stage1:
        log_policy, tok_policy = resp_logprob_detail(model, ["stage1", "stage2"], tok, prompt, resp, device, max_len)
        if cached_ref is not None:
            return log_policy, cached_ref[0], tok_policy, cached_ref[1]
        log_ref, tok_ref = resp_logprob_detail(model, "stage1", tok, prompt, resp, device, max_len)
    else:
        log_policy, tok_policy = resp_logprob_detail(model, ["stage2"], tok, prompt, resp, device, max_len)
        if cached_ref is not None:
            return log_policy, cached_ref[0], tok_policy, cached_ref[1]
        with model.disable_adapter():
            log_ref, tok_ref = resp_logprob_detail(model, [], tok, prompt, resp, device, max_len)
    return log_policy, log_ref, tok_policy, tok_ref


def score_pair(model, tok, row, device, max_len, has_stage1, cached=None):
    chosen_ref = rejected_ref = None
    if cached is not None:
        chosen_ref, rejected_ref = cached
    policy_chosen, ref_chosen, tok_policy_chosen, tok_ref_chosen = policy_and_reference_logprob(
        model, tok, row["prompt"], row["chosen"], device, max_len, has_stage1, chosen_ref
    )
    policy_rejected, ref_rejected, tok_policy_rejected, tok_ref_rejected = policy_and_reference_logprob(
        model, tok, row["prompt"], row["rejected"], device, max_len, has_stage1, rejected_ref
    )
    return {
        "pair_key": preference_pair_key(row),
        **preference_mia_statistics(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        ),
        "sequence_logprobs": {
            "policy_chosen": policy_chosen,
            "policy_rejected": policy_rejected,
            "ref_chosen": ref_chosen,
            "ref_rejected": ref_rejected,
        },
        "per_token_logprobs": {
            "policy_chosen": tok_policy_chosen,
            "policy_rejected": tok_policy_rejected,
            "ref_chosen": tok_ref_chosen,
            "ref_rejected": tok_ref_rejected,
        },
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


def prompt_hashes(rows):
    return {
        hashlib.sha256(str(row["prompt"]).strip().encode("utf-8")).hexdigest()
        for row in rows
    }


def assert_unique_pair_keys(records, label):
    keys = [record["pair_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} sample contains duplicate pair keys")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--member_jsonl", required=True)
    ap.add_argument("--nonmember_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument(
        "--per_token_jsonl",
        default=None,
        help="Per-token logprob artifact path. Defaults to <out_json stem>.per_token.jsonl.",
    )
    ap.add_argument(
        "--ref_logps",
        default=None,
        help="Eval reference cache from experiments/build_eval_ref_cache.py. When given, the "
             "base-model reference forward passes are served from it instead of recomputed, "
             "halving the forward passes in this script. Hash-validated per row; refuses a "
             "cache built for another purpose or another orientation.",
    )
    ap.add_argument("--subset_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument(
        "--heldout-verified-against",
        action="append",
        default=[],
        help="Additional training partition(s) the nonmember set must be disjoint from.",
    )
    ap.add_argument(
        "--i-know-this-is-not-heldout",
        action="store_true",
        help="Diagnostic only: accept an unverified nonmember set. The result is marked invalid.",
    )
    args = ap.parse_args()

    # Fail closed: the nonmember set must be shown disjoint from this run's training data. Members
    # are the training data, so verification is always possible here; --heldout-verified-against
    # adds any further partitions the model also saw.
    training_paths = [Path(args.member_jsonl)]
    training_paths += [Path(p) for p in args.heldout_verified_against]
    heldout_verification = assert_heldout_verified(
        Path(args.nonmember_jsonl),
        training_paths,
        role="nonmember",
        allow_unverified=args.i_know_this_is_not_heldout,
    )

    members = [json.loads(x) for x in open(args.member_jsonl, "r", encoding="utf-8") if x.strip()]
    nonmembers = [json.loads(x) for x in open(args.nonmember_jsonl, "r", encoding="utf-8") if x.strip()]
    overlap = prompt_hashes(members) & prompt_hashes(nonmembers)
    if overlap and not args.i_know_this_is_not_heldout:
        raise ValueError(
            f"Member/nonmember prompt overlap detected: {len(overlap)} prompt hashes"
        )
    sample_size = min(args.subset_size, len(members), len(nonmembers))
    if sample_size <= 0:
        raise ValueError("MIA requires non-empty member and nonmember datasets")
    members = deterministic_sample(members, sample_size, args.seed, "members")
    nonmembers = deterministic_sample(nonmembers, sample_size, args.seed, "nonmembers")

    device = torch.device("cuda")
    model, tok, has_stage1 = load_model(args.manifest, device)

    # Reference cache, if supplied. Loaded AFTER sampling so the returned lists are aligned
    # index-for-index with the sampled rows; load_reference_cache_detail validates every row's
    # prompt/chosen/rejected hashes, so a misalignment is a hard failure, not a silent shift.
    member_cached = nonmember_cached = None
    cache_stats = {"used": False}
    if args.ref_logps:
        # The manifest must be re-read here: load_model() parses it into a variable local to
        # itself, so there is no manifest in main()'s scope.
        _manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
        common = dict(
            expected_base_model=_manifest["base_model"],
            expected_scoring_dtype=REFERENCE_DTYPE,
            expected_truncation=TRUNCATION_MODE,
            expected_max_len=args.max_len,
            expected_rr_eps=None,
            expected_cache_role="eval_reference_true_orientation",
        )
        mc, mr, mcp, mrp = load_reference_cache_detail(args.ref_logps, members, **common)
        nc, nr, ncp, nrp = load_reference_cache_detail(args.ref_logps, nonmembers, **common)
        member_cached = [((mc[i], mcp[i]), (mr[i], mrp[i])) for i in range(len(members))]
        nonmember_cached = [((nc[i], ncp[i]), (nr[i], nrp[i])) for i in range(len(nonmembers))]
        cache_stats = {
            "used": True,
            "path": str(args.ref_logps),
            "member_hits": len(member_cached),
            "nonmember_hits": len(nonmember_cached),
            "hit_rate": 1.0,
            "reference_forwards_skipped": 2 * (len(members) + len(nonmembers)),
        }
        print(f"Reference cache: {cache_stats['member_hits']}+"
              f"{cache_stats['nonmember_hits']} pairs, "
              f"{cache_stats['reference_forwards_skipped']} reference forwards skipped",
              flush=True)

    member_records = []
    print("Scoring members...")
    for i, row in enumerate(members):
        member_records.append(score_pair(
            model, tok, row, device, args.max_len, has_stage1,
            member_cached[i] if member_cached else None))

    nonmember_records = []
    print("Scoring nonmembers...")
    for i, row in enumerate(nonmembers):
        nonmember_records.append(score_pair(
            model, tok, row, device, args.max_len, has_stage1,
            nonmember_cached[i] if nonmember_cached else None))
    assert_unique_pair_keys(member_records, "member")
    assert_unique_pair_keys(nonmember_records, "nonmember")

    attacks = {}
    for offset, attack in enumerate(MIA_STATISTICS):
        member_scores = [record[attack] for record in member_records]
        nonmember_scores = [record[attack] for record in nonmember_records]
        attacks[attack] = {
            **privacy_metrics(
                member_scores,
                nonmember_scores,
                args.seed + 1000 * offset,
                args.bootstrap_samples,
            ),
            "train_accuracy": preference_accuracy(member_scores),
            "heldout_accuracy": preference_accuracy(nonmember_scores),
            "mean_train_gap": sum(member_scores) / len(member_scores),
            "mean_heldout_gap": sum(nonmember_scores) / len(nonmember_scores),
        }
    calibrated = attacks["reference_calibrated_dpo_margin"]
    threshold = attacks["raw_policy_preference_gap"]
    calibrated_below = [
        metric
        for metric in ("auc", "tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr")
        if float(calibrated[metric]) < float(threshold[metric])
    ]
    member_deltas = [
        record["reference_calibrated_dpo_margin"] for record in member_records
    ]
    nonmember_deltas = [
        record["reference_calibrated_dpo_margin"] for record in nonmember_records
    ]

    out = {
        "manifest": args.manifest,
        "metric": "two_attack_preference_mia",
        "n_members": len(member_deltas),
        "n_nonmembers": len(nonmember_deltas),
        "mean_member_margin": sum(member_deltas) / len(member_deltas),
        "mean_nonmember_margin": sum(nonmember_deltas) / len(nonmember_deltas),
        # Attack-named keys. The bare `auc` / `mia_auc` below are the CALIBRATED attack's, while
        # `raw_mia_auc` elsewhere is a different attack — a naming collision that caused a
        # near-chance raw AUC to be read alongside the calibrated attack's TPR and reported as a
        # mathematical impossibility. Every metric now names the attack it belongs to.
        # See analysis_cpu/T1_g0_reconciliation.md.
        "auc_by_attack": {name: block.get("auc") for name, block in attacks.items()},
        "tpr_at_1pct_fpr_by_attack": {
            name: block.get("tpr_at_1pct_fpr") for name, block in attacks.items()
        },
        "tpr_at_0p1pct_fpr_by_attack": {
            name: block.get("tpr_at_0p1pct_fpr") for name, block in attacks.items()
        },
        "auc_ci_by_attack": {name: block.get("auc_ci") for name, block in attacks.items()},
        "primary_attack": "reference_calibrated_dpo_margin",
        # Retained for backward compatibility with existing readers. Both are the CALIBRATED
        # attack. Prefer the *_by_attack maps above; these are deprecated.
        "auc": calibrated["auc"],
        "mia_auc": calibrated["auc"],
        "deprecated_bare_metric_keys": [
            "auc",
            "mia_auc",
            "mean_member_margin",
            "mean_nonmember_margin",
        ],
        "attacks": attacks,
        "heldout_verification": heldout_verification,
        "attack_validity": {
            "passed": not calibrated_below,
            "calibrated_below_threshold_metrics": calibrated_below,
            "action": "continue" if not calibrated_below else "stop_and_report",
        },
        "reference_cache": cache_stats,
        "member_scores": member_records,
        "nonmember_scores": nonmember_records,
        "member_margins": member_deltas,
        "nonmember_margins": nonmember_deltas,
        "sampling": {
            "algorithm": "pair_key_sorted_namespaced_seed_v1",
            "seed": args.seed,
            "requested_subset_size": args.subset_size,
            "actual_subset_size": sample_size,
            "member_pair_keys_sha256": canonical_json_hash(
                [record["pair_key"] for record in member_records]
            ),
            "nonmember_pair_keys_sha256": canonical_json_hash(
                [record["pair_key"] for record in nonmember_records]
            ),
        },
        "provenance": {
            "manifest_sha256": file_sha256(args.manifest),
            "adapter_artifacts_sha256": adapter_artifacts_sha256(args.manifest),
            "member_jsonl_sha256": file_sha256(args.member_jsonl),
            "nonmember_jsonl_sha256": file_sha256(args.nonmember_jsonl),
            "scoring_config_sha256": canonical_json_hash(
                {
                    "max_len": args.max_len,
                    "precision": "float32",
                    "truncation": TRUNCATION_MODE,
                    "attacks": sorted(attacks),
                }
            ),
        },
        "precision": {
            "model_weights": "float32",
            "logprobs": LOGPROB_DTYPE,
            "reward_difference": LOGPROB_DTYPE,
        },
        "truncation": TRUNCATION_MODE,
        "max_len": args.max_len,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    output = Path(args.out_json)
    if "outputs" in output.resolve().parts:
        raise ValueError("refusing to write MIA artifact under outputs/")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Per-token scores are persisted beside the summary, one line per scored pair.
    # They are split out of the summary so out_json stays small enough to read.
    per_token_path = (
        Path(args.per_token_jsonl)
        if args.per_token_jsonl
        else output.with_name(output.stem + ".per_token.jsonl")
    )
    if "outputs" in per_token_path.resolve().parts:
        raise ValueError("refusing to write per-token artifact under outputs/")
    if per_token_path.exists():
        raise ValueError(f"refusing to overwrite existing per-token artifact: {per_token_path}")
    per_token_path.parent.mkdir(parents=True, exist_ok=True)
    per_token_temporary = per_token_path.with_suffix(per_token_path.suffix + ".tmp")
    token_counts = 0
    with per_token_temporary.open("w", encoding="utf-8") as handle:
        for split, records in (
            ("member", member_records),
            ("nonmember", nonmember_records),
        ):
            for record in records:
                per_token = record.pop("per_token_logprobs")
                token_counts += sum(len(values) for values in per_token.values())
                handle.write(
                    json.dumps(
                        {
                            "pair_key": record["pair_key"],
                            "split": split,
                            "per_token_logprobs": per_token,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    per_token_temporary.replace(per_token_path)
    out["per_token_artifact"] = {
        "path": str(per_token_path),
        "n_rows": len(member_records) + len(nonmember_records),
        "n_token_logprobs": token_counts,
        "sha256": file_sha256(per_token_path),
    }

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
