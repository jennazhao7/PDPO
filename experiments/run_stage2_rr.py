#!/usr/bin/env python3
"""Stage 2 -- randomized-response arms. Measures how RR on preference LABELS affects a
DIRECTIONAL membership statistic versus a SIGN-INVARIANT one.

Pre-registration: experiments/stage2_rr/PREREGISTRATION.md (recorded before any arm ran).

TWO THINGS THAT MUST NOT BE MISIMPLEMENTED
  1. RR flips the LABEL only. Done at data prep (experiments/prepare_stage2_rr.py); this runner
     merely points the trainer at the pre-flipped file. No noise, no dropping, no DP-SGD.
  2. The attack is scored against the TRUE, UNFLIPPED label. The MIA member set is the
     unflipped PKU v3 train split; the flipped file is trainer input ONLY.

ACCURACY CONVENTIONS -- both are recoverable from one evaluation pass, at no extra GPU cost.
  The evaluator is fed the TRUE-labelled train subsample and persists per-pair margins. For a
  row the model trained on with a flipped label, the training-objective margin is the negation of
  the true-label margin. So:
      train_acc_true    = mean(margin > 0)
      train_acc_flipped = mean(margin > 0 for unflipped rows) + mean(margin < 0 for flipped rows)
  train_acc_flipped is the fit-to-objective measure (how completely the model fit what it was
  actually trained on) and is the one used for fit-matching against the Stage 1 curve.
  Derived offline in analysis; both persisted.

CANARY GATE
  eps = 0 runs first. tanh(0) = 0, so the signed statistic MUST be at chance. After the first
  eps=0 checkpoint commits, the runner writes status `canary_gate` and EXITS. It will not run
  further arms until relaunched with --canary-cleared.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "stage2_debugging"):
    sys.path.insert(0, str(p))

BASE_MODEL = "Qwen/Qwen2.5-3B"
BETA, LR, SEED, GA, BSZ, MAX_LEN, LORA_R = 0.5, 2.5e-5, 42, 16, 1, 512, 16
EPOCHS = (1, 2, 3)
CHECKPOINT_PERCENTS = (25, 50, 75, 100)
# eps=0 first: correctness canary. Then the four PROPS-sourced values, ascending.
ARM_EPS = (0.0, 0.1, 0.5, 1.0, 2.0)

TRUE_TRAIN = "data/pku_saferlhf_secure_v3/train_pref.jsonl"   # attack member set + acc subsample
TEST = "data/pku_saferlhf_secure_v3/test_pref.jsonl"          # non-members, never flipped
RR_TRAIN = "data/stage2_rr/train_rr_eps{eps}_seed{seed}.jsonl"  # trainer input ONLY
CACHE = "experiments/ref_logps_stage2/rr_eps{eps}_seed{seed}_fp32.jsonl"
# ONE base-model reference cache serves eval for the whole sweep. Eval always scores the TRUE
# unflipped orientation, so unlike CACHE (per-arm, invalidated by the swap) this is arm-independent.
EVAL_CACHE = "experiments/ref_logps_stage2/eval_ref_true_fp32.jsonl"

OUT_ROOT = Path("experiments/stage2_rr")
LEDGER = OUT_ROOT / "ledger.jsonl"
STATUS = OUT_ROOT / "status.json"
SUBSAMPLE = 2000
BOOTSTRAP = 10000
EXPECTED = len(ARM_EPS) * len(EPOCHS) * len(CHECKPOINT_PERCENTS)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sync(path: Path, bucket: str) -> None:
    if not bucket or not path.exists():
        return
    try:
        subprocess.run(["gcloud", "storage", "cp", "-r", str(path),
                        f"{bucket.rstrip('/')}/{path.as_posix()}"],
                       check=True, capture_output=True, timeout=1800)
    except Exception as exc:  # noqa: BLE001 - sync failure must not lose committed progress
        print(f"[stage2] WARNING: sync failed {path}: {exc}", flush=True)


def write_status(status: str, progress: int, bucket: str, **extra) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "status": status, "progress": progress, "expected_cells": EXPECTED,
        "completed_cells": progress, "failure_signature": extra.pop("failure_signature", ""),
        "timestamp": utc(), **extra}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sync(STATUS, bucket)


def read_ledger() -> List[Dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def key_of(eps: float, epochs: int, pct: int) -> str:
    return f"rr_eps{eps}|e{epochs}|r{LORA_R}|p{pct:03d}"


def append_ledger(rec: Dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec, sort_keys=True) + "\n")
        h.flush()
        os.fsync(h.fileno())


def cfg_dir(eps: float, epochs: int) -> Path:
    return OUT_ROOT / f"rr_eps{eps}_e{epochs}_r{LORA_R}_seed{SEED}"


def ensure_cache(eps: float, bucket: str, dry: bool) -> str:
    """Reference cache per arm. The Stage 1 cache CANNOT be reused: load_reference_cache
    validates chosen_hash/rejected_hash, which the RR swap invalidates."""
    cache = Path(CACHE.format(eps=eps, seed=SEED))
    data = RR_TRAIN.format(eps=eps, seed=SEED)
    if cache.exists():
        # P1b added an rr_eps/cache_role assertion to load_reference_cache. A cache built before
        # that (the eps=0 one already in the bucket) carries neither, so the trainer would refuse
        # it -- while this early return would never rebuild it. That is a deadlock, so detect the
        # legacy shape here and regenerate rather than hand back a cache the trainer will reject.
        try:
            first = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            first = {}
        if first.get("rr_eps") == eps and first.get("cache_role") == "train_reference_arm_orientation":
            return str(cache)
        legacy = cache.with_suffix(cache.suffix + ".legacy_no_arm_tag")
        print(f"[stage2] {cache.name} predates the arm-identity assertion "
              f"(rr_eps={first.get('rr_eps')!r}); moving to {legacy.name} and rebuilding",
              flush=True)
        cache.rename(legacy)
    cmd = [sys.executable, "experiments/precompute_ref_logps_v3.py",
           "--data", data, "--out", str(cache), "--model", BASE_MODEL, "--max-len", str(MAX_LEN),
           "--rr-eps", str(eps)]
    print(f"[stage2] ref cache eps={eps}: {' '.join(cmd)}", flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        sync(cache, bucket)
    return str(cache)


def ensure_eval_cache(bucket: str, dry: bool) -> str:
    """Base-model reference cache for EVAL, built once for the entire sweep.

    Removes 18,000 of the 36,000 forward passes per checkpoint. Built at batch size 1 so the
    cached values come from the identical call path the evaluators use, which is what the
    equivalence gate certifies."""
    cache = Path(EVAL_CACHE)
    if cache.exists():
        return str(cache)
    cmd = [sys.executable, "experiments/build_eval_ref_cache.py",
           "--out", str(cache), "--model", BASE_MODEL, "--max-len", str(MAX_LEN),
           "--batch-size", "1"]
    print(f"[stage2] eval ref cache: {' '.join(cmd)}", flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        sync(cache, bucket)
    return str(cache)


def train(eps: float, epochs: int, dry: bool) -> Path:
    out = cfg_dir(eps, epochs)
    cmd = [sys.executable, "stage2_debugging/train_matched_cached.py",
           "--method", "mle", "--model", BASE_MODEL,
           "--data", RR_TRAIN.format(eps=eps, seed=SEED),
           "--ref_logps", CACHE.format(eps=eps, seed=SEED),
           "--out", str(out), "--epsilon", str(eps if eps > 0 else 1.0),
           "--expect_cache_eps", str(eps),
           "--beta", str(BETA), "--lr", str(LR), "--epochs", str(epochs),
           "--lora_r", str(LORA_R), "--bsz", str(BSZ), "--ga", str(GA),
           "--max_len", str(MAX_LEN), "--seed", str(SEED), "--fp32"]
    print(f"[stage2] train eps={eps} e{epochs}: {' '.join(cmd)}", flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    return out


def evaluate(eps: float, epochs: int, pct: int, dry: bool,
             eval_cache: Optional[str] = None) -> Dict[str, Any]:
    ck = cfg_dir(eps, epochs) / f"checkpoint_p{pct:03d}"
    manifest = ck / "M2_manifest.json"
    scratch = ck / "eval"
    scratch.mkdir(parents=True, exist_ok=True)
    ev, mia = scratch / "eval.json", scratch / "mia.json"
    sub = OUT_ROOT / "train_acc_subsample_true.jsonl"
    if not sub.exists() and not dry:
        from metric_primitives import deterministic_sample
        rows = [json.loads(l) for l in open(ROOT / TRUE_TRAIN, encoding="utf-8") if l.strip()]
        sample = deterministic_sample(rows, SUBSAMPLE, SEED, "stage2_train_acc_true")
        sub.parent.mkdir(parents=True, exist_ok=True)
        with sub.open("w", encoding="utf-8") as h:
            for r in sample:
                h.write(json.dumps(r, sort_keys=True) + "\n")
    # TRUE labels on both evaluators. The flipped file never appears here.
    acc_cmd = [sys.executable, "stage2_debugging/eval_preference_accuracy.py",
               "--manifest", str(manifest), "--test_jsonl", TEST,
               "--train_jsonl", str(sub), "--out_json", str(ev), "--max_len", str(MAX_LEN)]
    if eval_cache:
        acc_cmd += ["--ref_logps", eval_cache]
    mia_cmd = [sys.executable, "stage2_debugging/eval_privacy_audit.py",
               "--manifest", str(manifest), "--member_jsonl", TRUE_TRAIN,
               "--nonmember_jsonl", TEST, "--out_json", str(mia),
               "--subset_size", str(SUBSAMPLE), "--seed", str(SEED),
               "--max_len", str(MAX_LEN), "--bootstrap_samples", str(BOOTSTRAP)]
    if eval_cache:
        mia_cmd += ["--ref_logps", eval_cache]
    for c in (acc_cmd, mia_cmd):
        print(f"[stage2] {' '.join(c)}", flush=True)
        if not dry:
            subprocess.run(c, check=True, cwd=str(ROOT))
    if dry:
        return {"dry_run": True}

    a = json.loads(ev.read_text(encoding="utf-8"))
    m = json.loads(mia.read_text(encoding="utf-8"))
    st = json.loads((ck / "checkpoint_state.json").read_text(encoding="utf-8"))
    sp = a.get("splits", {})

    def acc(split, stat):
        b = sp.get(split, {}).get(stat)
        return b.get("accuracy") if isinstance(b, dict) else None

    tr = acc("train", "reference_calibrated_dpo_margin")
    ho = acc("heldout", "reference_calibrated_dpo_margin")
    return {
        "loss": st.get("loss"), "step": st.get("step"),
        "planned_total_steps": st.get("planned_total_steps"),
        "train_accuracy_true": tr, "heldout_accuracy_calibrated": ho,
        "train_heldout_gap_true": (tr - ho) if (tr is not None and ho is not None) else None,
        "heldout_accuracy_raw": acc("heldout", "raw_policy_preference_gap"),
        "train_accuracy_raw_true": acc("train", "raw_policy_preference_gap"),
        "n_heldout": a.get("n"), "n_members": m.get("n_members"),
        "n_nonmembers": m.get("n_nonmembers"),
        # Non-members defining each FPR threshold -- required beside every TPR.
        "n_nonmembers_defining_fpr_1pct": int(round(0.01 * (m.get("n_nonmembers") or 0))),
        "mia_attacks": m.get("attacks"),
        "heldout_verification": m.get("heldout_verification"),
        "per_token_artifact": m.get("per_token_artifact"),
        "eval_json": str(ev), "mia_json": str(mia),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", ""))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--canary-cleared", action="store_true",
                    help="Proceed past the eps=0 canary gate. Set only after the signed AUC "
                         "at the first eps=0 checkpoint has been confirmed at chance.")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from heldout_guard import assert_heldout_verified
    assert_heldout_verified(ROOT / TEST, [ROOT / TRUE_TRAIN], role="stage2 non-member")

    done = {r["key"] for r in read_ledger()}
    progress = len(done)
    write_status("running", progress, args.bucket, canary_cleared=args.canary_cleared)

    # Built once, before any arm. It needs the accuracy eval's train subsample, which
    # evaluate() would otherwise write on its first call, so build_eval_ref_cache generates that
    # file itself when absent using the identical deterministic_sample call.
    eval_cache = ensure_eval_cache(args.bucket, args.dry_run)

    for eps in ARM_EPS:
        ensure_cache(eps, args.bucket, args.dry_run)
        for epochs in EPOCHS:
            keys = [key_of(eps, epochs, p) for p in CHECKPOINT_PERCENTS]
            if all(k in done for k in keys):
                continue
            d = cfg_dir(eps, epochs)
            if not all((d / f"checkpoint_p{p:03d}" / "M2_manifest.json").exists()
                       for p in CHECKPOINT_PERCENTS):
                try:
                    train(eps, epochs, args.dry_run)
                except subprocess.CalledProcessError as exc:
                    write_status("failed", progress, args.bucket,
                                 failure_signature=f"train:eps{eps}:e{epochs}:rc{exc.returncode}")
                    raise
            for pct in CHECKPOINT_PERCENTS:
                k = key_of(eps, epochs, pct)
                if k in done:
                    continue
                try:
                    metrics = evaluate(eps, epochs, pct, args.dry_run, eval_cache)
                except Exception as exc:  # noqa: BLE001
                    write_status("failed", progress, args.bucket,
                                 failure_signature=f"eval:{k}:{type(exc).__name__}",
                                 traceback=traceback.format_exc()[-2000:])
                    raise
                if args.dry_run:
                    continue
                append_ledger({"key": k, "arm": "rr", "eps": eps,
                               "gamma": 1.0 / (1.0 + pow(2.718281828459045, eps)),
                               "epochs": epochs, "lora_r": LORA_R, "percent": pct,
                               "beta": BETA, "lr": LR, "seed": SEED,
                               "trainer_data": RR_TRAIN.format(eps=eps, seed=SEED),
                               "attack_member_jsonl": TRUE_TRAIN,
                               "scored_against": "TRUE unflipped label",
                               "metrics": metrics, "timestamp": utc()})
                done.add(k)
                progress += 1
                sync(LEDGER, args.bucket)
                sync(cfg_dir(eps, epochs) / f"checkpoint_p{pct:03d}", args.bucket)
                write_status("running", progress, args.bucket)
                print(f"[stage2] committed {k} ({progress} done)", flush=True)

                if eps == 0.0 and not args.canary_cleared:
                    at = (metrics.get("mia_attacks") or {}).get(
                        "reference_calibrated_dpo_margin") or {}
                    write_status("canary_gate", progress, args.bucket,
                                 canary_cell=k,
                                 canary_signed_auc=at.get("auc"),
                                 canary_signed_auc_ci=at.get("auc_ci"),
                                 note="eps=0 predicts signed AUC at chance (tanh(0)=0). "
                                      "STOPPED for confirmation. Relaunch with --canary-cleared.")
                    print(f"[stage2] CANARY GATE: {k} signed AUC={at.get('auc')}. STOPPING.",
                          flush=True)
                    return 0

    write_status("complete", progress, args.bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
