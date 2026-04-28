#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

SCRIPT="${ROOT_DIR}/stage2_debugging/stage2_d3_pku/run_s2.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

MAIL_USER="jzhao7@nd.edu"

submit_job() {
  local job_name="$1" method="$2" base_model="$3" stage1_adapter="$4" dataset_path="$5" out_dir="$6"
  
  local qs_out="${out_dir}/qsub_${job_name}.out"
  local qs_err="${out_dir}/qsub_${job_name}.err"
  mkdir -p "${out_dir}"
  
  echo "[INFO] Submitting ${job_name}..."
  
  qsub -terse \
    -q gpu@@jung_gpu \
    -l gpu_card=1 \
    -pe smp 4 \
    -l h_rt=04:00:00 \
    -M "${MAIL_USER}" \
    -m bea \
    -N "${job_name}" \
    -o "${qs_out}" \
    -e "${qs_err}" \
    -v ROOT_DIR="${ROOT_DIR}",METHOD="${method}",BASE_MODEL="${base_model}",STAGE1_ADAPTER="${stage1_adapter}",DATASET_PATH="${dataset_path}",OUT_DIR="${out_dir}" \
    "${SCRIPT}"
}

# 1. D1+D2 M1 -> SB_Fresh on D3
submit_job "sb_d1d2_d3" "sb_fresh" "Qwen/Qwen2.5-3B" \
  "${ROOT_DIR}/stage2_debugging/stage1/results_base_d1d2_eps1.0_seed42/pku_base_d1d2_eps1.0_s42" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d3_disjoint_rr_flipped_pku_eps1.0_seed42.jsonl" \
  "${ROOT_DIR}/stage2_debugging/stage2_d3_pku/sb_fresh_d1d2_m1_on_d3"

# 2. D1+D2 M1 -> MAP_Retrain on D3
submit_job "map_d1d2_d3" "map_retrain" "Qwen/Qwen2.5-3B" \
  "${ROOT_DIR}/stage2_debugging/stage1/results_base_d1d2_eps1.0_seed42/pku_base_d1d2_eps1.0_s42" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d3_disjoint_rr_flipped_pku_eps1.0_seed42.jsonl" \
  "${ROOT_DIR}/stage2_debugging/stage2_d3_pku/map_retrain_d1d2_m1_on_d3"

# 3. Instruct M1 -> SB_Fresh on D2
submit_job "sb_inst_d2" "sb_fresh" "Qwen/Qwen2.5-3B-Instruct" \
  "${ROOT_DIR}/stage2_debugging/stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_pku_eps1.0_seed42.jsonl" \
  "${ROOT_DIR}/stage2_debugging/stage2_d3_pku/sb_fresh_instruct_m1_on_d2"

# 4. Instruct M1 -> MAP_Retrain on D2
submit_job "map_inst_d2" "map_retrain" "Qwen/Qwen2.5-3B-Instruct" \
  "${ROOT_DIR}/stage2_debugging/stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_pku_eps1.0_seed42.jsonl" \
  "${ROOT_DIR}/stage2_debugging/stage2_d3_pku/map_retrain_instruct_m1_on_d2"

echo "[DONE] All 4 jobs submitted."
