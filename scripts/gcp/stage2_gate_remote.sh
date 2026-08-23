#!/usr/bin/env bash
# On-VM driver for the equivalence gate. Builds the eval reference cache, re-evaluates the
# already-committed checkpoint rr_eps0.0|e1|r16|p025 through the OPTIMIZED path into a scratch
# directory, and compares against the committed artifacts.
#
# Writes only under experiments/stage2_gate/ and experiments/ref_logps_stage2/. It never touches
# the committed cell's eval/ directory, so a failed gate cannot corrupt the ledger's artifacts.
set -euo pipefail
ROOT="${HOME}/PDPO"; OUT_DIR="${ROOT}/experiments/stage2_gate"
BUCKET_URI="${GCS_BUCKET}"; [[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
GCS_OUT="${BUCKET_URI}/experiments/stage2_gate"
CELL="rr_eps0.0_e1_r16_seed42/checkpoint_p025"
mkdir -p "${OUT_DIR}"; cd "${ROOT}"
log() { echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/gate.log"; }
publish() { gcloud storage cp -r "${OUT_DIR}" "${BUCKET_URI}/experiments/" >/dev/null 2>&1 || true; }
( set +e +o pipefail; while true; do sleep 120; publish; done ) >/dev/null 2>&1 &

on_error() { rc=$?; trap - ERR; log "GATE FAILED rc=${rc}"; publish; exit "${rc}"; }
trap on_error ERR

log "installing deps"
# The deep-learning image's system python3 ships without pip; stage2_remote.sh bootstraps it and
# this script must too. Output is NOT redirected to /dev/null -- doing so hid the real error and
# left only "GATE FAILED rc=1" to debug from.
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "image python3 lacks pip; installing python3-pip"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
fi
python3 -m pip install -r requirements.txt 2>&1 | tail -5 | tee -a "${OUT_DIR}/gate.log"
python3 -c "import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'L4' in n,n;print(n)"
log "parity tests (the guarantee that cached reference values are substitutable)"
python3 -m unittest discover -s tests -p 'test_ref_logprob_parity.py'

log "restoring committed artifacts for the baseline comparison"
gcloud storage cp "${BUCKET_URI}/experiments/stage2_rr/train_acc_subsample_true.jsonl" \
  "${ROOT}/experiments/stage2_rr/" 2>/dev/null || true
gcloud storage cp -r "${BUCKET_URI}/experiments/stage2_rr/rr_eps0.0_e1_r16_seed42" \
  "${ROOT}/experiments/stage2_rr/" 2>/dev/null || true
gcloud storage cp -r "${BUCKET_URI}/experiments/ref_logps_stage2" "${ROOT}/experiments/" 2>/dev/null || true

BASE="${ROOT}/experiments/stage2_rr/${CELL}"
[[ -f "${BASE}/eval/mia.json" ]] || { log "baseline mia.json absent; cannot gate"; exit 2; }

log "building the eval reference cache (batch size 1, bit-identical call path)"
CACHE="${ROOT}/experiments/ref_logps_stage2/eval_ref_true_fp32.jsonl"
if [[ ! -f "${CACHE}" ]]; then
  /usr/bin/time -v python3 experiments/build_eval_ref_cache.py --out "${CACHE}" --batch-size 1 \
    2>&1 | tail -20 | tee -a "${OUT_DIR}/cache_build.log"
  gcloud storage cp "${CACHE}" "${BUCKET_URI}/experiments/ref_logps_stage2/" || true
else
  log "cache already present"
fi
publish

log "re-evaluating ${CELL} through the OPTIMIZED path"
CAND="${OUT_DIR}/candidate"; mkdir -p "${CAND}"
S=$(date -u +%s)
python3 stage2_debugging/eval_preference_accuracy.py \
  --manifest "${BASE}/M2_manifest.json" \
  --test_jsonl data/pku_saferlhf_secure_v3/test_pref.jsonl \
  --train_jsonl experiments/stage2_rr/train_acc_subsample_true.jsonl \
  --out_json "${CAND}/eval.json" --max_len 512 --ref_logps "${CACHE}"
A=$(date -u +%s); log "accuracy eval took $((A-S))s"
python3 stage2_debugging/eval_privacy_audit.py \
  --manifest "${BASE}/M2_manifest.json" \
  --member_jsonl data/pku_saferlhf_secure_v3/train_pref.jsonl \
  --nonmember_jsonl data/pku_saferlhf_secure_v3/test_pref.jsonl \
  --out_json "${CAND}/mia.json" --subset_size 2000 --seed 42 --max_len 512 \
  --bootstrap_samples 10000 --ref_logps "${CACHE}"
B=$(date -u +%s); log "privacy audit took $((B-A))s ; TOTAL OPTIMIZED EVAL $((B-S))s"
printf '{"accuracy_eval_seconds":%d,"privacy_audit_seconds":%d,"total_seconds":%d,"baseline_total_seconds":7375}\n' \
  "$((A-S))" "$((B-A))" "$((B-S))" > "${OUT_DIR}/timing.json"
publish

log "running the equivalence gate"
set +e
python3 experiments/equivalence_gate.py \
  --baseline-mia "${BASE}/eval/mia.json" --candidate-mia "${CAND}/mia.json" \
  --baseline-eval "${BASE}/eval/eval.json" --candidate-eval "${CAND}/eval.json" \
  --out-json "${OUT_DIR}/GATE_RESULT.json" 2>&1 | tee -a "${OUT_DIR}/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
log "gate rc=${GATE_RC} (0=PASS)"
publish
exit 0
