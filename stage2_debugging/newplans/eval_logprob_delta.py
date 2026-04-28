"""
eval_logprob_delta.py
---------------------
Principled preference-preservation eval using log-prob delta.

WHAT IT MEASURES
----------------
For each test pair (x, y_w, y_l):

  delta_base  = log p_base(y_w | x)  - log p_base(y_l | x)
  delta_model = log p_model(y_w | x) - log p_model(y_l | x)

  shift = delta_model - delta_base

A positive shift means the adapter increased the chosen/rejected gap
relative to base — i.e., it reinforced the preference signal.

PRIMARY METRICS
---------------
1. mean_shift        : mean(delta_model - delta_base)
   → How much the adapter moved the gap on average.
     Higher = better preference reinforcement.

2. pref_accuracy     : frac(delta_model > 0)
   → What fraction of pairs the model ranks correctly (chosen > rejected).
     Ties broken at 0.5 for count purposes.

3. pref_accuracy_base: frac(delta_base > 0)
   → Same but for base model. Sanity check / lower bound.

4. shift_on_correct  : mean shift where base already got it right
5. shift_on_flipped  : mean shift where base got it wrong
   → Decomposition: are we helping on hard cases or just easy ones?

6. margin_gain_pct   : pct of pairs where delta_model > delta_base
   → Did the adapter improve the margin on the majority of pairs?

USAGE
-----
# Single model vs base:
python eval_logprob_delta.py \
    --model_path ./models/sb_fresh_truthy_eps0.5_seed42 \
    --base_model_path meta-llama/Llama-3.2-3B \
    --test_data ./data/truthy_test_100.jsonl \
    --output ./results/sb_fresh_truthy_eps0.5_seed42_delta.json

# Compare multiple models (comparison table):
python eval_logprob_delta.py \
    --model_paths ./models/mle_fresh_truthy_eps0.5_seed42 \
                  ./models/map_retrain_truthy_eps0.5_seed42 \
                  ./models/sb_fresh_truthy_eps0.5_seed42 \
    --model_names mle_fresh map_retrain sb_fresh \
    --base_model_path meta-llama/Llama-3.2-3B \
    --test_data ./data/truthy_test_100.jsonl \
    --output ./results/truthy_eps0.5_comparison.json \
    --print_table

# With explicit epsilon (enables per-pair noise-adjusted analysis):
python eval_logprob_delta.py \
    --model_path ./models/sb_fresh_truthy_eps0.5_seed42 \
    --base_model_path meta-llama/Llama-3.2-3B \
    --test_data ./data/truthy_test_100.jsonl \
    --epsilon 0.5 \
    --output ./results/sb_fresh_truthy_eps0.5_seed42_delta.json

DATA FORMAT
-----------
JSONL, one pair per line. Required fields:
  prompt    : str
  chosen    : str
  rejected  : str

Optional fields (used for richer analysis if present):
  was_flipped : bool   — whether RR flipped this pair's label
  pair_id     : str    — for cross-run pair matching
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PreferencePairDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _encode_pair(prompt: str, response: str, tokenizer, max_length: int = 512):
    """
    Encode (prompt, response) and return token ids + a mask that marks
    only the response tokens (used when computing per-token log-probs).
    """
    full_text = prompt + response
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    n_prompt_tokens = prompt_enc["input_ids"].shape[1]

    input_ids = enc["input_ids"]          # (1, T)
    attention_mask = enc["attention_mask"] # (1, T)

    # Response mask: True only for response tokens (positions n_prompt_tokens onwards)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[0, n_prompt_tokens:] = True

    return input_ids, attention_mask, response_mask


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """
    Return the mean per-token log-probability of the response tokens.
    Using mean (not sum) makes the metric comparable across different
    response lengths.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    response_mask = response_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (1, T, V)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[0, :-1, :]          # (T-1, V)
    shift_labels = input_ids[0, 1:]            # (T-1,)
    shift_mask   = response_mask[0, 1:]        # (T-1,)

    if shift_mask.sum() == 0:
        return float("nan")

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs[torch.arange(len(shift_labels)), shift_labels]  # (T-1,)

    mean_lp = token_lp[shift_mask].mean().item()
    return mean_lp


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    pairs: list[dict],
    max_length: int = 512,
    batch_desc: str = "",
) -> dict:
    """
    For each pair, compute delta = log p(y_w|x) - log p(y_l|x).
    Returns a dict of raw per-pair values and aggregate metrics.
    """
    deltas = []
    lp_chosen_list = []
    lp_rejected_list = []
    was_flipped_list = []

    for pair in tqdm(pairs, desc=batch_desc, leave=False):
        prompt   = pair["prompt"]
        chosen   = pair["chosen"]
        rejected = pair["rejected"]

        ids_w, mask_w, rmask_w = _encode_pair(prompt, chosen,   tokenizer, max_length)
        ids_l, mask_l, rmask_l = _encode_pair(prompt, rejected, tokenizer, max_length)

        lp_w = compute_logprob(model, ids_w, mask_w, rmask_w)
        lp_l = compute_logprob(model, ids_l, mask_l, rmask_l)

        delta = lp_w - lp_l
        deltas.append(delta)
        lp_chosen_list.append(lp_w)
        lp_rejected_list.append(lp_l)
        was_flipped_list.append(pair.get("was_flipped", None))

    deltas = np.array(deltas)
    pref_acc = float(np.mean(deltas > 0))
    mean_delta = float(np.mean(deltas))

    return {
        "deltas":           deltas.tolist(),
        "lp_chosen":        lp_chosen_list,
        "lp_rejected":      lp_rejected_list,
        "was_flipped":      was_flipped_list,
        "pref_accuracy":    pref_acc,
        "mean_delta":       mean_delta,
        "std_delta":        float(np.std(deltas)),
        "n_pairs":          len(pairs),
    }


# ---------------------------------------------------------------------------
# Comparison / shift analysis
# ---------------------------------------------------------------------------

def compute_shifts(base_result: dict, model_result: dict, epsilon: Optional[float] = None) -> dict:
    """
    Given base and model results, compute all shift-based metrics.
    """
    base_deltas  = np.array(base_result["deltas"])
    model_deltas = np.array(model_result["deltas"])
    shifts       = model_deltas - base_deltas

    base_correct  = base_deltas > 0          # base already ranked correctly
    base_wrong    = ~base_correct

    metrics = {
        # Core shift metrics
        "mean_shift":        float(np.mean(shifts)),
        "std_shift":         float(np.std(shifts)),
        "margin_gain_pct":   float(np.mean(shifts > 0)),   # frac pairs improved

        # Accuracy comparison
        "pref_accuracy_base":  base_result["pref_accuracy"],
        "pref_accuracy_model": model_result["pref_accuracy"],
        "accuracy_delta":      model_result["pref_accuracy"] - base_result["pref_accuracy"],

        # Decomposition: easy vs hard pairs
        "mean_shift_on_base_correct": float(np.mean(shifts[base_correct]))
            if base_correct.sum() > 0 else float("nan"),
        "mean_shift_on_base_wrong":   float(np.mean(shifts[base_wrong]))
            if base_wrong.sum() > 0 else float("nan"),
        "n_base_correct": int(base_correct.sum()),
        "n_base_wrong":   int(base_wrong.sum()),

        # Raw delta stats
        "mean_delta_base":  base_result["mean_delta"],
        "mean_delta_model": model_result["mean_delta"],

        "n_pairs": model_result["n_pairs"],
    }

    # If epsilon provided, compute noise-theory-calibrated expected shift
    if epsilon is not None:
        # Under RR with epsilon, fraction of labels flipped = gamma_eps
        gamma_eps = 1.0 / (1.0 + math.exp(epsilon))
        # A perfect denoiser should recover (1 - 2*gamma_eps) fraction of signal
        # Expected mean_shift if we perfectly undo flips = (1-2*gamma)*|mean_base_delta|
        denoising_potential = (1.0 - 2.0 * gamma_eps) * abs(base_result["mean_delta"])
        metrics["epsilon"]             = epsilon
        metrics["gamma_eps"]           = gamma_eps
        metrics["flip_rate_theory"]    = gamma_eps
        metrics["denoising_potential"] = denoising_potential
        # How much of the potential did we recover?
        if abs(denoising_potential) > 1e-6:
            metrics["recovery_fraction"] = metrics["mean_shift"] / denoising_potential
        else:
            metrics["recovery_fraction"] = float("nan")

    # Flipped-pair analysis (only if was_flipped flags present)
    was_flipped = model_result["was_flipped"]
    if any(f is not None for f in was_flipped):
        flipped_mask  = np.array([bool(f) for f in was_flipped])
        clean_mask    = ~flipped_mask
        metrics["mean_shift_on_flipped_pairs"] = float(np.mean(shifts[flipped_mask]))   \
            if flipped_mask.sum() > 0 else float("nan")
        metrics["mean_shift_on_clean_pairs"]   = float(np.mean(shifts[clean_mask]))     \
            if clean_mask.sum() > 0 else float("nan")
        metrics["n_flipped_pairs"] = int(flipped_mask.sum())
        metrics["n_clean_pairs"]   = int(clean_mask.sum())

    return metrics


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_single_result(name: str, base_res: dict, model_res: dict, shift_metrics: dict):
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")
    print(f"  Pairs evaluated      : {model_res['n_pairs']}")
    print(f"  Pref accuracy (base) : {shift_metrics['pref_accuracy_base']:.3f}")
    print(f"  Pref accuracy (model): {shift_metrics['pref_accuracy_model']:.3f}  "
          f"(Δ = {shift_metrics['accuracy_delta']:+.3f})")
    print(f"  Mean delta (base)    : {shift_metrics['mean_delta_base']:+.4f}")
    print(f"  Mean delta (model)   : {shift_metrics['mean_delta_model']:+.4f}")
    print(f"  Mean shift           : {shift_metrics['mean_shift']:+.4f}  ±{shift_metrics['std_shift']:.4f}")
    print(f"  Margin gain (% pairs): {shift_metrics['margin_gain_pct']:.3f}")
    print(f"  Shift on base-correct: {shift_metrics['mean_shift_on_base_correct']:+.4f}  "
          f"(n={shift_metrics['n_base_correct']})")
    print(f"  Shift on base-wrong  : {shift_metrics['mean_shift_on_base_wrong']:+.4f}  "
          f"(n={shift_metrics['n_base_wrong']})")

    if "epsilon" in shift_metrics:
        print(f"\n  -- Noise analysis (ε={shift_metrics['epsilon']}) --")
        print(f"  Flip rate (theory)   : {shift_metrics['flip_rate_theory']:.3f}")
        print(f"  Denoising potential  : {shift_metrics['denoising_potential']:+.4f}")
        rf = shift_metrics.get("recovery_fraction", float("nan"))
        print(f"  Recovery fraction    : {rf:.3f}" if not math.isnan(rf) else
              f"  Recovery fraction    : n/a")

    if "n_flipped_pairs" in shift_metrics:
        print(f"\n  -- Per-flip-status breakdown --")
        print(f"  Shift on flipped pairs: {shift_metrics['mean_shift_on_flipped_pairs']:+.4f}  "
              f"(n={shift_metrics['n_flipped_pairs']})")
        print(f"  Shift on clean pairs  : {shift_metrics['mean_shift_on_clean_pairs']:+.4f}  "
              f"(n={shift_metrics['n_clean_pairs']})")


def print_comparison_table(results: list[dict]):
    """
    results is a list of dicts, each with 'name' and 'shift_metrics'.
    """
    print("\n" + "="*90)
    print(f"  {'Method':<22}  {'Acc(base)':>9}  {'Acc(model)':>10}  {'Acc Δ':>7}  "
          f"{'Mean shift':>11}  {'Margin gain':>11}")
    print("  " + "-"*86)
    for r in results:
        m = r["shift_metrics"]
        print(f"  {r['name']:<22}  {m['pref_accuracy_base']:>9.3f}  "
              f"{m['pref_accuracy_model']:>10.3f}  {m['accuracy_delta']:>+7.3f}  "
              f"{m['mean_shift']:>+11.4f}  {m['margin_gain_pct']:>11.3f}")
    print("="*90)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, base_path: str, device: str):
    """
    Load a PEFT adapter on top of base, or just the base if model_path==base_path.
    Falls back to pure HF load if peft is not installed or path has no adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        base.eval()
        if model_path != base_path:
            model = PeftModel.from_pretrained(base, model_path)
            model.eval()
        else:
            model = base
    except (ImportError, OSError):
        # No PEFT or no adapter checkpoint — just load the path directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Log-prob delta eval for DP-Pref-LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model specification — either single or multiple
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path",  type=str,
                       help="Path to single PEFT adapter dir or HF model.")
    group.add_argument("--model_paths", type=str, nargs="+",
                       help="Paths to multiple models for comparison table.")

    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                        help="Display names for --model_paths (same order). "
                             "Defaults to directory basenames.")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="HF model path/name for the base model.")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to JSONL test file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write JSON results.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Privacy budget used during training (enables noise analysis).")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for prompt+response.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--print_table", action="store_true",
                        help="Print comparison table (only with --model_paths).")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load test pairs
    pairs = []
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} test pairs from {args.test_data}")

    # --- Evaluate base model once (shared across all comparisons) ---
    print(f"\nEvaluating base model: {args.base_model_path}")
    base_model, base_tokenizer = load_model_and_tokenizer(
        args.base_model_path, args.base_model_path, device
    )
    base_result = evaluate_model(
        base_model, base_tokenizer, pairs,
        max_length=args.max_length,
        batch_desc="base model",
    )
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Base pref_accuracy = {base_result['pref_accuracy']:.3f}")
    print(f"  Base mean_delta    = {base_result['mean_delta']:+.4f}")

    # --- Build list of (model_path, model_name) pairs ---
    if args.model_path:
        model_specs = [(args.model_path, Path(args.model_path).name)]
    else:
        names = args.model_names or [Path(p).name for p in args.model_paths]
        if len(names) != len(args.model_paths):
            print("ERROR: --model_names must have same length as --model_paths", file=sys.stderr)
            sys.exit(1)
        model_specs = list(zip(args.model_paths, names))

    # --- Evaluate each model ---
    all_results = []
    for model_path, model_name in model_specs:
        print(f"\nEvaluating: {model_name}  ({model_path})")
        model, tokenizer = load_model_and_tokenizer(
            model_path, args.base_model_path, device
        )
        model_result = evaluate_model(
            model, tokenizer, pairs,
            max_length=args.max_length,
            batch_desc=model_name,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        shift_metrics = compute_shifts(base_result, model_result, epsilon=args.epsilon)
        print_single_result(model_name, base_result, model_result, shift_metrics)

        all_results.append({
            "name":         model_name,
            "model_path":   model_path,
            "shift_metrics": shift_metrics,
            "raw":          {
                "model": {k: v for k, v in model_result.items() if k != "deltas"},
                "base":  {k: v for k, v in base_result.items()  if k != "deltas"},
            },
            # Keep per-pair deltas for downstream analysis
            "per_pair": {
                "base_deltas":  base_result["deltas"],
                "model_deltas": model_result["deltas"],
                "shifts":       (np.array(model_result["deltas"]) -
                                 np.array(base_result["deltas"])).tolist(),
                "was_flipped":  model_result["was_flipped"],
            },
        })

    # Comparison table if multiple models
    if len(all_results) > 1 and args.print_table:
        print_comparison_table(all_results)

    # --- Save ---
    output = {
        "config": {
            "base_model":  args.base_model_path,
            "test_data":   args.test_data,
            "n_pairs":     len(pairs),
            "epsilon":     args.epsilon,
            "max_length":  args.max_length,
        },
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
