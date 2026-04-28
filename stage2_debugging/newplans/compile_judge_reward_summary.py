#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iso_mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def parse_record(eval_report_path: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": str(eval_report_path),
        "acquired_datetime": iso_mtime(eval_report_path),
    }
    data = load_json(eval_report_path) or {}

    rel = eval_report_path.as_posix().split("/lora/eval/")[-1]
    parts = rel.split("/")
    rec["group"] = parts[0] if parts else "unknown"
    rec["model_family"] = parts[1] if len(parts) > 2 else "unknown"
    rec["comparison"] = parts[2] if len(parts) > 3 else (parts[-2] if len(parts) >= 2 else "unknown")

    rec["model_a"] = data.get("model_a")
    rec["model_b"] = data.get("model_b")
    rec["n_prompts"] = data.get("n_prompts")
    rec["seed"] = data.get("seed")

    gpt = data.get("gpt4_judge", {}) if isinstance(data.get("gpt4_judge"), dict) else {}
    rec["judge_model"] = gpt.get("judge_model")
    rec["judge_wins_a"] = gpt.get("wins_a")
    rec["judge_wins_b"] = gpt.get("wins_b")
    rec["judge_ties"] = gpt.get("ties")
    rec["judge_win_rate_a"] = gpt.get("win_rate_a")
    rec["judge_win_rate_b"] = gpt.get("win_rate_b")
    rec["judge_tie_rate"] = gpt.get("tie_rate")
    rec["judge_win_rate_a_excl_ties"] = gpt.get("win_rate_a_excl_ties")

    reward_block = data.get("reward_bench", {}) if isinstance(data.get("reward_bench"), dict) else {}
    reward_item = None
    if reward_block:
        # Keep first reward-model entry (all current reports use one)
        reward_item = next(iter(reward_block.values()))
        if not isinstance(reward_item, dict):
            reward_item = None
    rec["reward_model"] = get_nested(reward_item or {}, "reward_model")
    rec["reward_score_a_mean"] = get_nested(reward_item or {}, "score_a_mean")
    rec["reward_score_b_mean"] = get_nested(reward_item or {}, "score_b_mean")
    rec["reward_wins_a"] = get_nested(reward_item or {}, "reward_wins_a")
    rec["reward_wins_b"] = get_nested(reward_item or {}, "reward_wins_b")
    rec["reward_ties"] = get_nested(reward_item or {}, "reward_ties")
    rec["reward_win_rate_a"] = get_nested(reward_item or {}, "reward_win_rate_a")
    rec["reward_win_rate_b"] = get_nested(reward_item or {}, "reward_win_rate_b")
    rec["reward_tie_rate"] = get_nested(reward_item or {}, "reward_tie_rate")

    gpt_summary_path = eval_report_path.with_name("gpt4_summary.json")
    reward_summary_candidates = list(eval_report_path.parent.glob("reward_summary_*.json"))
    rec["has_gpt_summary_file"] = gpt_summary_path.exists()
    rec["has_reward_summary_file"] = len(reward_summary_candidates) > 0
    return rec


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["acquired_datetime"], reverse=True)
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows_sorted:
        by_group.setdefault(r["group"], []).append(r)

    lines: List[str] = []
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines.append("# GPT-as-Judge and Reward-Model Results Summary")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append("- Acquisition date basis: filesystem modification time of each `eval_report.json`.")
    lines.append(f"- Total comparisons found: **{len(rows_sorted)}**")
    lines.append("")

    for grp, grp_rows in sorted(by_group.items()):
        lines.append(f"## {grp}")
        lines.append("")
        lines.append("| Acquired | Family | Comparison | A vs B | N | GPT judge A win | RM A win |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for r in grp_rows:
            lines.append(
                "| "
                f"{r['acquired_datetime']} | "
                f"{r.get('model_family','-')} | "
                f"{r.get('comparison','-')} | "
                f"{r.get('model_a','-')} vs {r.get('model_b','-')} | "
                f"{fmt(r.get('n_prompts'),0)} | "
                f"{fmt(r.get('judge_win_rate_a'))} | "
                f"{fmt(r.get('reward_win_rate_a'))} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile GPT-as-judge and reward-model comparison results.")
    parser.add_argument("--repo_root", type=str, default="/users/jzhao7/PDPO")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/users/jzhao7/PDPO/stage2_debugging/newplans/crossdata_eval/judge_reward_inventory",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_reports = sorted((repo_root / "lora" / "eval").glob("**/eval_report.json"))
    rows = [parse_record(p) for p in eval_reports]
    rows = sorted(rows, key=lambda r: r["acquired_datetime"], reverse=True)

    out_json = out_dir / "judge_reward_summary.json"
    out_csv = out_dir / "judge_reward_summary.csv"
    out_md = out_dir / "judge_reward_summary.md"

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)

    print(f"[OK] Wrote {len(rows)} records")
    print(f"[OK] {out_json}")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
