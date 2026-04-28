#!/usr/bin/env python3
"""
aggregate_panel.py
==================
Aggregate judge JSONL files for a single (model_A vs model_B) comparison.

For each judge, and then for the panel majority, computes:
  - Win rates (A wins / B wins / ties) after position-swap debiasing
  - Panel agreement rate
  - Panel majority vote win rate
  - Bootstrap 95% CI on win rates (1000 resamples)
  - Per-prompt verdicts JSONL for error analysis

Input layout (from judge_panel.py):
  --input_dir results/pku_eps0.5/sb_vs_map/
    con_j.jsonl
    gpt4o.jsonl     (optional)
    gemini.jsonl    (optional)
    meta.json

Output in same directory:
  con_j.json        per-judge summary
  gpt4o.json
  gemini.json
  panel_summary.json   overall panel summary
  per_prompt.jsonl     per-prompt multi-judge verdicts

Usage:
  python aggregate_panel.py --input_dir results/pku_eps0.5/sb_vs_map/
"""

import argparse
import json
import os
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_win_rate(verdicts: list[str], target: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """
    Bootstrap 95% CI for the win rate of `target` ('A' or 'B').
    Returns {mean, ci_lo, ci_hi} rounded to 4 decimal places.
    """
    rng = random.Random(seed)
    n = len(verdicts)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None}

    point_est = verdicts.count(target) / n
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choices(verdicts, k=n)
        boot_means.append(sample.count(target) / n)

    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return {
        "mean": round(point_est, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise_judge(records: list[dict], judge_name: str) -> dict:
    verdicts = [r["final_verdict"] for r in records]
    n = len(verdicts)
    n_a = verdicts.count("A")
    n_b = verdicts.count("B")
    n_tie = verdicts.count("TIE")

    model_a = records[0].get("model_a", "A") if records else "A"
    model_b = records[0].get("model_b", "B") if records else "B"

    ci_a = bootstrap_win_rate(verdicts, "A")
    ci_b = bootstrap_win_rate(verdicts, "B")
    ci_tie = bootstrap_win_rate(verdicts, "TIE")

    return {
        "judge": judge_name,
        "model_a": model_a,
        "model_b": model_b,
        "n": n,
        "a_wins": n_a,
        "b_wins": n_b,
        "ties": n_tie,
        "a_win_rate": round(n_a / n, 4) if n else None,
        "b_win_rate": round(n_b / n, 4) if n else None,
        "tie_rate": round(n_tie / n, 4) if n else None,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "ci_tie": ci_tie,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Majority vote
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(votes: list[str]) -> str:
    """Return 'A', 'B', or 'TIE' based on simple majority. Ties → 'TIE'."""
    n_a = votes.count("A")
    n_b = votes.count("B")
    if n_a > n_b:
        return "A"
    elif n_b > n_a:
        return "B"
    else:
        return "TIE"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate panel judge results.")
    ap.add_argument("--input_dir", required=True,
                    help="Directory with <judge>.jsonl files from judge_panel.py.")
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="Bootstrap resamples for CI (default: 1000).")
    args = ap.parse_args()

    input_dir = args.input_dir
    judge_names = ["con_j", "gpt4o", "gemini"]

    # Load available judge files
    judge_records: dict[str, list[dict]] = {}
    for jname in judge_names:
        path = os.path.join(input_dir, f"{jname}.jsonl")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        judge_records[jname] = rows
        print(f"[Load] {jname}: {len(rows)} records", flush=True)

    if not judge_records:
        print("[ERROR] No judge JSONL files found in", input_dir)
        return

    # Per-judge summaries
    summaries = {}
    for jname, records in judge_records.items():
        summary = summarise_judge(records, jname)
        summaries[jname] = summary
        out_path = os.path.join(input_dir, f"{jname}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[{jname}] A={summary['a_wins']} B={summary['b_wins']} TIE={summary['ties']} "
              f"(A win rate={summary['a_win_rate']})", flush=True)

    # Build per-prompt multi-judge view, aligned by prompt_id
    # Use the first judge as the reference set of prompt_ids
    ref_judge = list(judge_records.keys())[0]
    ref_records = judge_records[ref_judge]

    # Index all judges by prompt_id
    by_pid: dict[str, dict[str, str]] = defaultdict(dict)
    for jname, records in judge_records.items():
        for r in records:
            by_pid[r["prompt_id"]][jname] = r["final_verdict"]

    # Also keep reference fields
    ref_by_pid = {r["prompt_id"]: r for r in ref_records}

    per_prompt = []
    n_all_agree = 0
    panel_verdicts = []

    available_judges = list(judge_records.keys())

    for pid, judge_verdicts in by_pid.items():
        votes = [judge_verdicts.get(j, None) for j in available_judges if j in judge_verdicts]
        votes_clean = [v for v in votes if v is not None]
        panel_v = majority_vote(votes_clean) if votes_clean else "TIE"
        panel_verdicts.append(panel_v)

        all_agree = len(set(votes_clean)) == 1 if votes_clean else False
        if all_agree:
            n_all_agree += 1

        ref = ref_by_pid.get(pid, {})
        row = {
            "prompt_id": pid,
            "prompt": ref.get("prompt", ""),
            "response_a": ref.get("response_a", ""),
            "response_b": ref.get("response_b", ""),
            "model_a": ref.get("model_a", "A"),
            "model_b": ref.get("model_b", "B"),
            "judge_verdicts": judge_verdicts,
            "panel_majority_verdict": panel_v,
            "all_judges_agree": all_agree,
        }
        per_prompt.append(row)

    # Per-prompt JSONL
    per_prompt_path = os.path.join(input_dir, "per_prompt.jsonl")
    with open(per_prompt_path, "w", encoding="utf-8") as f:
        for row in per_prompt:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[per_prompt] written to {per_prompt_path}", flush=True)

    # Panel summary
    n_total = len(panel_verdicts)
    model_a = ref_records[0].get("model_a", "A") if ref_records else "A"
    model_b = ref_records[0].get("model_b", "B") if ref_records else "B"

    panel_summary = {
        "model_a": model_a,
        "model_b": model_b,
        "judges_available": available_judges,
        "n": n_total,
        "panel_agreement_rate": round(n_all_agree / n_total, 4) if n_total else None,
        "panel_majority": {
            "a_wins": panel_verdicts.count("A"),
            "b_wins": panel_verdicts.count("B"),
            "ties": panel_verdicts.count("TIE"),
            "a_win_rate": round(panel_verdicts.count("A") / n_total, 4) if n_total else None,
            "b_win_rate": round(panel_verdicts.count("B") / n_total, 4) if n_total else None,
            "tie_rate": round(panel_verdicts.count("TIE") / n_total, 4) if n_total else None,
            "ci_a": bootstrap_win_rate(panel_verdicts, "A", n_boot=args.n_boot),
            "ci_b": bootstrap_win_rate(panel_verdicts, "B", n_boot=args.n_boot),
        },
        "per_judge": summaries,
    }

    panel_path = os.path.join(input_dir, "panel_summary.json")
    with open(panel_path, "w", encoding="utf-8") as f:
        json.dump(panel_summary, f, indent=2)

    print(f"\n[Panel Summary]")
    print(f"  Agreement rate: {panel_summary['panel_agreement_rate']}")
    print(f"  {model_a} wins: {panel_summary['panel_majority']['a_wins']} "
          f"({panel_summary['panel_majority']['a_win_rate']})")
    print(f"  {model_b} wins: {panel_summary['panel_majority']['b_wins']} "
          f"({panel_summary['panel_majority']['b_win_rate']})")
    print(f"  TIEs: {panel_summary['panel_majority']['ties']}")
    print(f"[Done] panel summary → {panel_path}", flush=True)


if __name__ == "__main__":
    main()
