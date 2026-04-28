#!/bin/bash
#$ -N pdpo_truthy_eval
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 4
#$ -l h_rt=2:00:00

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

TEST_JSONL="stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"
EVAL_PY="stage2_debugging/eval_preference_accuracy.py"

mkdir -p outputs_new_models/evals

run_eval() {
    local method_dir=$1
    local name=$2
    local gpu=$3
    local manifest="${method_dir}/M2_manifest.json"
    local out_json="outputs_new_models/evals/${name}_eval.json"
    local log="outputs_new_models/logs/eval_${name}.log"
    echo "Starting eval for ${name} on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=${gpu} python -u ${EVAL_PY} \
        --manifest "${manifest}" \
        --test_jsonl "${TEST_JSONL}" \
        --out_json "${out_json}" > "${log}" 2>&1
    echo "[DONE] Eval for ${name} saved to ${out_json}"
}

run_eval "outputs_new_models/stage2_cdpo/truthy_eps0.5_s42" "cdpo_truthy_eps0.5" 0 &
run_eval "outputs_new_models/stage2_cdpo/truthy_eps1.0_s42" "cdpo_truthy_eps1.0" 1 &
run_eval "outputs_new_models/stage2_rdpo/truthy_eps0.5_s42" "rdpo_truthy_eps0.5" 2 &
run_eval "outputs_new_models/stage2_rdpo/truthy_eps1.0_s42" "rdpo_truthy_eps1.0" 3 &

wait
echo "All configurations evaluated successfully."
