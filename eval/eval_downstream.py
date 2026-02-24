#!/usr/bin/env python3
"""
Downstream evaluation: safety classification, RewardBench, helpfulness ranking.
Loads models via M2 manifest (same pattern as eval_compare.py).
"""
import argparse
import json
import os

# Reuse stage2 loading from eval_compare
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="M2_manifest.json path")
    ap.add_argument("--task", choices=["safety", "rewardbench", "helpfulness"], default="safety")
    ap.add_argument("--data_dir", default="data/pku_saferlhf_secure")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    # TODO: Implement safety classification (PKU harm categories)
    # TODO: Implement RewardBench (allenai/reward-bench)
    # TODO: Implement helpfulness ranking (AlpacaEval subset)
    print("eval_downstream.py: stub. Implement safety, rewardbench, helpfulness tasks.")
    print(f"  manifest={args.manifest} task={args.task}")


if __name__ == "__main__":
    main()
