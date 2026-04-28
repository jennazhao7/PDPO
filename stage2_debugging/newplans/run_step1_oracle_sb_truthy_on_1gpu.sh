#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
ORACLE_DELTA="${ORACLE_DELTA:-10.0}"
TAU="${TAU:-1.0}"

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/truthy_eps${EPS}_s${SEED}}"
D2_DATA="${D2_DATA:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl}"
ORACLE_LABELS="${ORACLE_LABELS:-}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/step1_oracle_sb/results_truthy_eps${EPS}_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"
OUT_ORACLE="${OUT_ROOT}/oracle_sb_stacked_truthy_eps${EPS}_s${SEED}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}"

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[FATAL] conda not found in PATH"
  exit 2
fi

cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false

for f in "${S1_ADAPTER}/adapter_model.safetensors" "${D2_DATA}" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done
if [[ -n "${ORACLE_LABELS}" ]]; then
  [ -f "${ORACLE_LABELS}" ] || { echo "[FATAL] ORACLE_LABELS set but missing: ${ORACLE_LABELS}"; exit 2; }
fi

python - <<'PY'
import torch
from datasets import load_dataset
import os
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
d2 = load_dataset("json", data_files=os.environ["D2_DATA"])["train"]
print("d2_rows:", len(d2))
assert len(d2) > 0, "D2 empty"
PY

# If ORACLE_LABELS is not provided, reconstruct flip indicators directly from D2 RR output.
# This is exact when D2 has stable `id` from rr_stream_flip.
if [[ -z "${ORACLE_LABELS}" ]]; then
  ORACLE_LABELS="${OUT_ROOT}/oracle_labels_from_d2_rr.jsonl"
  export D2_DATA ORACLE_LABELS EPS SEED
  python - <<'PY'
import json, math, os, hashlib

d2 = os.environ["D2_DATA"]
out = os.environ["ORACLE_LABELS"]
eps = float(os.environ["EPS"])
seed = int(os.environ["SEED"])
q = 1.0 / (math.exp(eps) + 1.0)

rows = [json.loads(x) for x in open(d2, "r", encoding="utf-8") if x.strip()]
if not rows:
    raise RuntimeError(f"D2 empty: {d2}")
if "id" not in rows[0]:
    raise RuntimeError(
        "D2 rows do not contain 'id'. Cannot exactly reconstruct RR flips from D2 alone. "
        "Please pass --ORACLE_LABELS explicitly."
    )

def stable_uniform(seed: int, example_id: str) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(str(example_id).encode("utf-8"))
    value = int.from_bytes(h.digest(), byteorder="big", signed=False)
    return value / 2**64

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
flips = 0
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        ex_id = str(r["id"])
        rr_flipped = stable_uniform(seed, ex_id) < q
        flips += int(rr_flipped)
        rec = {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "id": r["id"],
            "flipped": bool(rr_flipped),
            "source": "reconstructed_from_d2_rr",
            "epsilon": eps,
            "seed": seed,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[Oracle-Reconstruct] wrote {len(rows)} rows to {out}; flipped={flips}/{len(rows)} q={q:.6f}")
PY
fi

export ORACLE_LABELS

# Validate oracle labels (provided or reconstructed)
python - <<'PY'
from datasets import load_dataset
import os
d2 = load_dataset("json", data_files=os.environ["D2_DATA"])["train"]
lbl = load_dataset("json", data_files=os.environ["ORACLE_LABELS"])["train"]
print("oracle_rows:", len(lbl))
assert len(lbl) == len(d2), f"oracle rows {len(lbl)} != d2 rows {len(d2)}"
assert "flipped" in lbl.column_names, f"oracle_labels missing 'flipped'; cols={lbl.column_names}"
PY

echo "[INFO] Running Step-1 Oracle SB (stacked architecture)..."
CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_oracle_soft_bayes.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --oracle_labels "${ORACLE_LABELS}" \
  --out "${OUT_ORACLE}" \
  --epsilon "${EPS}" \
  --tau "${TAU}" \
  --oracle_delta "${ORACLE_DELTA}" \
  --beta 0.5 \
  --lr 2.5e-5 \
  --seed "${SEED}" \
  --epochs 3 \
  --max_steps -1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bsz 1 \
  --ga 16 \
  --bf16 \
  2>&1 | tee "${LOG_DIR}/train_oracle_sb_truthy_s${SEED}.log"

MANIFEST="${OUT_ORACLE}/M2_manifest.json"
[ -f "${MANIFEST}" ] || { echo "[FATAL] Missing manifest: ${MANIFEST}"; exit 3; }

echo "[INFO] Evaluating oracle SB..."
CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${MANIFEST}" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/oracle_sb_truthy_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  2>&1 | tee "${LOG_DIR}/eval_oracle_sb_truthy_s${SEED}.log"

export OUT_ROOT EVAL_DIR EPS SEED
python - <<'PY'
import json, os
out_root = os.environ["OUT_ROOT"]
eval_dir = os.environ["EVAL_DIR"]
eps = os.environ["EPS"]
seed = os.environ["SEED"]
oracle_path = os.path.join(eval_dir, f"oracle_sb_truthy_eps{eps}_s{seed}_eval.json")
baseline_path = f"/users/jzhao7/PDPO/stage2_debugging/stage2_truthy/results_eps{eps}_seed{seed}/eval/mle_dpo_truthy_eps{eps}_s{seed}_eval.json"

summary = {}
if os.path.exists(oracle_path):
    d = json.load(open(oracle_path))
    summary["Oracle-SB-Stacked"] = d
if os.path.exists(baseline_path):
    d = json.load(open(baseline_path))
    summary["MLE-DPO-Stacked-baseline"] = d

delta = None
if "Oracle-SB-Stacked" in summary and "MLE-DPO-Stacked-baseline" in summary:
    delta = summary["Oracle-SB-Stacked"]["accuracy"] - summary["MLE-DPO-Stacked-baseline"]["accuracy"]
summary["oracle_minus_mle_accuracy"] = delta

json_path = os.path.join(out_root, "STEP1_ORACLE_SB_SUMMARY.json")
md_path = os.path.join(out_root, "STEP1_ORACLE_SB_SUMMARY.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Step1 Oracle SB Summary (truthy, eps={eps}, seed={seed})\n\n")
    f.write(f"- Oracle eval: `{oracle_path}`\n")
    f.write(f"- Baseline MLE eval: `{baseline_path}`\n")
    if delta is None:
        f.write("- Delta accuracy: N/A (missing one eval file)\n")
    else:
        f.write(f"- Delta accuracy (oracle - mle): **{delta:.4f}**\n")
print("Wrote", json_path)
print("Wrote", md_path)
PY

echo "[DONE] Step1 Oracle SB completed: ${OUT_ROOT}"

