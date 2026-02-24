#!/usr/bin/env python3
"""
Multi-seed runner + aggregator for DP-LoRA.

Runs train_dp_lora.py once per seed, then aggregates convergence summaries
and produces cross-seed statistics + an aggregate plot with mean +/- 1 std.

Usage:
    python run_multiseed.py \\
        --seeds 0 1 2 3 4 \\
        --base_output_dir outputs/dp_multiseed \\
        -- --model_id gpt2-medium --dataset_path d1.jsonl \\
           --dp --noise_multiplier 1.0 --target_modules c_attn,c_proj,c_fc \\
           --max_steps 5

Everything after '--' is forwarded to train_dp_lora.py (with --seed and
--output_dir injected per run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(values)}


# ---------------------------------------------------------------------------
# Aggregate convergence curves across seeds
# ---------------------------------------------------------------------------

def interpolate_curve(steps: List[int], values: List[float], grid: List[int]) -> List[float]:
    """Linearly interpolate (steps, values) onto a common grid."""
    result: List[float] = []
    for g in grid:
        if g <= steps[0]:
            result.append(values[0])
        elif g >= steps[-1]:
            result.append(values[-1])
        else:
            # Find bracketing interval
            for i in range(len(steps) - 1):
                if steps[i] <= g <= steps[i + 1]:
                    frac = (g - steps[i]) / max(1, steps[i + 1] - steps[i])
                    result.append(values[i] + frac * (values[i + 1] - values[i]))
                    break
            else:
                result.append(values[-1])
    return result


def aggregate_curves(
    run_dirs: List[str],
    metric_key: str = "eval_pairwise_accuracy",
) -> Dict[str, Any]:
    """Collect per-seed eval curves, interpolate, compute mean/std."""
    all_steps: List[List[int]] = []
    all_vals: List[List[float]] = []

    for rd in run_dirs:
        mpath = os.path.join(rd, "metrics.jsonl")
        if not os.path.exists(mpath):
            continue
        metrics = read_jsonl(mpath)
        pts = [(m["step"], m[metric_key]) for m in metrics if metric_key in m]
        if not pts:
            continue
        steps, vals = zip(*pts)
        all_steps.append(list(steps))
        all_vals.append(list(vals))

    if not all_steps:
        return {}

    # Build common step grid (union of all step points, sorted)
    grid_set = set()
    for s in all_steps:
        grid_set.update(s)
    grid = sorted(grid_set)

    # Interpolate each curve onto grid
    interp_matrix: List[List[float]] = []
    for steps, vals in zip(all_steps, all_vals):
        interp_matrix.append(interpolate_curve(steps, vals, grid))

    mat = np.array(interp_matrix)  # [n_seeds, n_grid]
    mean_curve = mat.mean(axis=0).tolist()
    std_curve = mat.std(axis=0).tolist()

    return {
        "grid": grid,
        "mean": mean_curve,
        "std": std_curve,
        "n_seeds": len(interp_matrix),
        "metric_key": metric_key,
    }


def plot_aggregate(
    curves: Dict[str, Any],
    output_path: str,
    train_curves: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot mean eval metric +/- 1 std band."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AggPlot] matplotlib not installed; skipping aggregate_plot.png.")
        return

    if not curves or "grid" not in curves:
        print("[AggPlot] no curves to plot.")
        return

    grid = curves["grid"]
    mean = np.array(curves["mean"])
    std = np.array(curves["std"])
    mk = curves.get("metric_key", "eval_metric")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(grid, mean, "b-", linewidth=2, label=f"mean {mk}")
    ax1.fill_between(grid, mean - std, mean + std, alpha=0.25, color="b", label="$\\pm 1$ std")
    ax1.set_xlabel("global_step")
    ax1.set_ylabel(mk)
    ax1.legend(loc="lower right")
    ax1.set_title(f"Aggregate ({curves['n_seeds']} seeds)")

    if train_curves and "grid" in train_curves:
        ax2 = ax1.twinx()
        tg = train_curves["grid"]
        tm = np.array(train_curves["mean"])
        ts = np.array(train_curves["std"])
        ax2.plot(tg, tm, "r-", alpha=0.5, linewidth=1, label="mean train loss")
        ax2.fill_between(tg, tm - ts, tm + ts, alpha=0.1, color="r")
        ax2.set_ylabel("train loss", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[AggPlot] saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Split argv at '--'
    try:
        sep = sys.argv.index("--")
    except ValueError:
        sep = len(sys.argv)

    runner_argv = sys.argv[1:sep]
    train_argv = sys.argv[sep + 1:] if sep < len(sys.argv) else []

    ap = argparse.ArgumentParser(description="Multi-seed DP-LoRA runner + aggregator.")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seeds.")
    ap.add_argument("--base_output_dir", type=str, required=True)
    ap.add_argument("--skip_training", action="store_true",
                    help="Skip training; only aggregate existing results.")
    ap.add_argument("--metric_key", type=str, default="eval_pairwise_accuracy",
                    help="Metric key for convergence analysis.")
    args = ap.parse_args(runner_argv)

    os.makedirs(args.base_output_dir, exist_ok=True)
    run_dirs: List[str] = []

    # -----------------------------------------------------------------------
    # Run training per seed
    # -----------------------------------------------------------------------
    if not args.skip_training:
        # Import main from train_dp_lora
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from train_dp_lora import main as train_main

        for seed in args.seeds:
            run_dir = os.path.join(args.base_output_dir, f"seed_{seed}")
            run_dirs.append(run_dir)

            # Build argv for this seed
            seed_argv = list(train_argv)
            # Remove any existing --seed / --output_dir / --auto_seed_dir from forwarded args
            filtered: List[str] = []
            skip_next = False
            for i, a in enumerate(seed_argv):
                if skip_next:
                    skip_next = False
                    continue
                if a in ("--seed", "--output_dir"):
                    skip_next = True
                    continue
                if a == "--auto_seed_dir":
                    continue
                filtered.append(a)
            seed_argv = filtered + ["--seed", str(seed), "--output_dir", run_dir]

            print("\n" + "=" * 70)
            print(f"  SEED {seed}  ->  {run_dir}")
            print("=" * 70 + "\n")

            ret = train_main(seed_argv)
            if ret != 0:
                print(f"[ERROR] seed {seed} returned {ret}")
    else:
        for seed in args.seeds:
            run_dirs.append(os.path.join(args.base_output_dir, f"seed_{seed}"))

    # -----------------------------------------------------------------------
    # Aggregate convergence summaries
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATION")
    print("=" * 70)

    summaries: List[Dict] = []
    for rd in run_dirs:
        cpath = os.path.join(rd, "convergence_summary.json")
        if os.path.exists(cpath):
            with open(cpath, "r") as f:
                summaries.append(json.load(f))

    agg: Dict[str, Any] = {"n_seeds": len(summaries), "seeds": args.seeds}
    for key in ["best_eval_metric", "final_eval_metric", "auc_eval_metric_vs_steps", "steps_to_95pct_best"]:
        vals = [s[key] for s in summaries if s.get(key) is not None]
        agg[key] = mean_std(vals)

    # Also aggregate final summary.json fields
    for rd in run_dirs:
        spath = os.path.join(rd, "summary.json")
        if os.path.exists(spath):
            with open(spath, "r") as f:
                s = json.load(f)
            for key in ["train_time_seconds", "epsilon_spent"]:
                if key not in agg:
                    agg[key] = {"values": []}
                if s.get(key) is not None:
                    agg[key]["values"].append(s[key])

    for key in ["train_time_seconds", "epsilon_spent"]:
        if key in agg and "values" in agg[key]:
            vals = agg[key].pop("values")
            agg[key] = mean_std(vals)

    write_json(os.path.join(args.base_output_dir, "aggregate_summary.json"), agg)
    print(f"[Agg] aggregate_summary.json written")
    for k, v in sorted(agg.items()):
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Aggregate plot
    # -----------------------------------------------------------------------
    eval_curves = aggregate_curves(run_dirs, args.metric_key)
    train_curves = aggregate_curves(run_dirs, "loss") if eval_curves else None
    plot_aggregate(
        eval_curves,
        os.path.join(args.base_output_dir, "aggregate_plot.png"),
        train_curves=train_curves,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
