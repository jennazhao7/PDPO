#!/usr/bin/env bash
# On-VM driver for E9 track B1: generation only, no judging.
# Writes only under experiments/e9_utility/. Touches nothing the Stage 2 sweep reads or writes.
set -euo pipefail
ROOT="${HOME}/PDPO"; OUT_DIR="${ROOT}/experiments/e9_utility"
BUCKET_URI="${GCS_BUCKET}"; [[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
mkdir -p "${OUT_DIR}"; cd "${ROOT}"
log() { echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/e9.log"; }
publish() { gcloud storage cp -r "${OUT_DIR}" "${BUCKET_URI}/experiments/" >/dev/null 2>&1 || true; }
( set +e +o pipefail; while true; do sleep 120; publish; done ) >/dev/null 2>&1 &

on_error() { rc=$?; trap - ERR; log "E9 GENERATION FAILED rc=${rc}"; publish; exit "${rc}"; }
trap on_error ERR

log "installing deps"
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "image python3 lacks pip; installing python3-pip"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
fi
python3 -m pip install -r requirements.txt 2>&1 | tail -3 | tee -a "${OUT_DIR}/e9.log"
python3 -c "import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'L4' in n,n;print(n)"

log "restoring Stage 1 checkpoints for the three E9 models"
for c in pku_e1_r16_seed42/checkpoint_p075 pku_e2_r16_seed42/checkpoint_p050 pku_e3_r16_seed42/checkpoint_p100; do
  mkdir -p "${ROOT}/experiments/stage1_operating_point/$(dirname "$c")"
  gcloud storage cp -r "${BUCKET_URI}/experiments/stage1_operating_point/${c}" \
    "${ROOT}/experiments/stage1_operating_point/$(dirname "$c")/" 2>&1 | tail -1
done
find "${ROOT}/experiments/stage1_operating_point" -name adapter_config.json | tee -a "${OUT_DIR}/e9.log"

log "generating (FP32, greedy, 300 held-out prompts x 4 models)"
S=$(date -u +%s)
python3 experiments/run_e9_generation.py --out-dir "${OUT_DIR}" --n-prompts 300 \
  --max-new-tokens 256 --batch-size 8 2>&1 | tee -a "${OUT_DIR}/e9.log"
E=$(date -u +%s)
log "generation complete in $((E-S))s"
publish
log "E9 GENERATION DONE"
publish
exit 0
