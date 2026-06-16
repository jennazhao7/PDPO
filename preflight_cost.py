#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import config
from experiments import aggregate


def ledger_spend(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row.get("est_cost", 0.0) or 0.0)
            except ValueError:
                continue
    return total


def result_path(result_root: Path, cell: Dict) -> Path:
    eps = aggregate.slug_eps(float(cell["eps"]))
    cid = f"{cell['id']}_{cell['method']}_{cell['dataset']}_eps{eps}_seed{cell['seed']}"
    return result_root / str(cell["id"]) / f"{cid}.json"


def valid_existing(path: Path) -> bool:
    result, _error = aggregate.load_result(path) if path.exists() else (None, "missing")
    return result is not None


def remaining_cells(manifest: Dict, result_root: Path, force: bool) -> List[Dict]:
    cells = aggregate.expand_manifest(manifest)
    if force:
        return cells
    return [cell for cell in cells if not valid_existing(result_path(result_root, cell))]


def parse_args() -> argparse.Namespace:
    root = aggregate.repo_root()
    ap = argparse.ArgumentParser(description="Refuse launches that exceed the PDPO credit ceiling.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--result-root", default=str(root / "experiments/run_queue"))
    ap.add_argument("--ledger", default=str(root / "cost_ledger.csv"))
    ap.add_argument("--card", default=config.PREFERRED_CARD)
    ap.add_argument("--hours-per-cell", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="Estimate as if all cells rerun.")
    ap.add_argument("--yes", action="store_true", help="Non-interactive approval after budget check.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config.validate_runtime_config(require_bucket=True)
    card = args.card.upper()
    card_cfg = config.get_card_config(card)
    manifest = aggregate.load_manifest(Path(args.manifest))
    remaining = remaining_cells(manifest, Path(args.result_root), args.force)
    spend_to_date = ledger_spend(Path(args.ledger))
    projected_gpu_hours = len(remaining) * args.hours_per_cell
    projected_cost = projected_gpu_hours * card_cfg.rough_spot_usd_per_hour
    projected_total = spend_to_date + projected_cost

    print(json.dumps({
        "card": card,
        "spot_rate_usd_per_hour_est": card_cfg.rough_spot_usd_per_hour,
        "queued_cells": len(remaining),
        "hours_per_cell": args.hours_per_cell,
        "projected_gpu_hours": projected_gpu_hours,
        "spend_to_date": spend_to_date,
        "projected_cost": projected_cost,
        "projected_total": projected_total,
        "credit_ceiling_usd": config.CREDIT_CEILING_USD,
    }, indent=2))

    if projected_total > config.CREDIT_CEILING_USD:
        raise SystemExit(
            f"REFUSING launch: projected total ${projected_total:.2f} exceeds "
            f"ceiling ${config.CREDIT_CEILING_USD:.2f}."
        )
    if args.yes:
        return 0
    typed = input('Type "yes" to proceed with this launch estimate: ').strip()
    if typed != "yes":
        raise SystemExit("Launch cancelled by preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
