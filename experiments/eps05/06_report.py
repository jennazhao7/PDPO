#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict


def _latest_master(results_root: str) -> str:
    candidates = sorted(glob.glob(os.path.join(results_root, "master_summary_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No master_summary_*.json under {results_root}")
    return candidates[-1]


def _pick_reward(entry: Dict[str, Any]) -> Dict[str, float]:
    keys = [k for k in entry.keys() if k.startswith("reward_") and k.endswith("_mean_a")]
    if not keys:
        return {"reward_a": float("nan"), "reward_b": float("nan")}
    k = sorted(keys)[0]
    kb = k.replace("_mean_a", "_mean_b")
    return {"reward_a": entry.get(k, float("nan")), "reward_b": entry.get(kb, float("nan"))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact eps05 leaderboard from eval summary.")
    ap.add_argument("--results_root", default="lora/eval/results_eps05")
    ap.add_argument("--summary_json", default=None, help="Optional explicit master summary path.")
    ap.add_argument("--out_json", default="lora/eval/results_eps05/leaderboard_compact.json")
    args = ap.parse_args()

    summary_path = args.summary_json or _latest_master(args.results_root)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = []
    for comp_key, entry in summary.get("comparisons", {}).items():
        reward = _pick_reward(entry)
        rows.append(
            {
                "comparison": comp_key,
                "model_a": entry.get("model_a"),
                "model_b": entry.get("model_b"),
                "n_prompts": entry.get("n_prompts"),
                "judge_win_a": entry.get("gpt4_win_a"),
                "judge_win_b": entry.get("gpt4_win_b"),
                "judge_tie": entry.get("gpt4_tie"),
                "reward_mean_a": reward["reward_a"],
                "reward_mean_b": reward["reward_b"],
            }
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_summary": summary_path,
                "n_rows": len(rows),
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote compact leaderboard: {args.out_json}")


if __name__ == "__main__":
    main()
