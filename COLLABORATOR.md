# Collaborator Setup: Running the SB Dominance Plan

This doc explains what to push to GitHub and what a collaborator needs to run the 6 to-dos from the SB Dominance Proof Plan.

---

## 1. What NOT to Push (GitHub-Unfriendly)

| Item | Size | Why |
|------|------|-----|
| `outputs_openllama_tonight/` | ~3 GB | LoRA adapters, logs. Contains `.safetensors` files 15–153 MB each. **GitHub limit: 100 MB per file** — many exceed it. |
| `outputs/`, `outputs_eps05/` | ~1–2 GB | Same: trained adapters. |
| `data/pku_saferlhf_secure/raw_hf_dataset/` | ~114 MB | HuggingFace dataset snapshot. Regeneratable. |
| `*.log`, `__pycache__/` | varies | Build artifacts. |

**LoRAs are stored as:** `outputs_openllama_tonight/{stage1,stage2_mle_fix,stage2_softbayes}/<model>_stage*_eps1.0_seed*/`  
Each dir has `adapter_model.safetensors`, `adapter_config.json`, and (for stage2) `M2_manifest.json` with absolute paths to stage1 + stage2.

**Solution for sharing LoRAs:** Push adapters to **Hugging Face Hub** (not GitHub). Use `push_model_to_hf.py` or `huggingface-cli upload`. Reference by HF repo ID in manifests.

---

## 2. Minimal Push Checklist (GitHub)

### Must push (code + config)

```
# Core training & eval
lora/train_stage1_rr_lora.py          # Stage 1 RR SFT
lora/train_stage2_lora.py             # Stage 2 MLE
lora/train_stage2_soft_bayes.py       # Stage 2 Soft-Bayes
lora/preprocessing/rr_stream_flip.py  # RR flipping
lora/preprocessing/prepare_d1_d2.sh   # (or run_data_prep_task.sh)
lora/verify_stage2_stack.py           # Model loading helper

eval/eval_compare.py                  # Reward-only comparison
eval/check_rr_flip_rate.py            # Step 2A (uses audit file)
eval/check_sb_logit_shift.py          # Step 2B
eval/eval_preference_accuracy.py      # Step 3
eval/eval_downstream.py               # Step 5 (stub)

experiments/core_result_tonight/01_secure_pku_saferlhf.py
experiments/core_result_tonight/run_data_prep_task.sh
experiments/core_result_tonight/run_stage1_task.sh
experiments/core_result_tonight/run_stage2_mle_task.sh
experiments/core_result_tonight/run_stage2_sb_task.sh

requirements.txt                      # Add peft>=0.10
```

### Optional but useful

```
data/pku_saferlhf_secure/train_pref.jsonl   # 36 MB — under 100 MB limit
data/pku_saferlhf_secure/test_pref.jsonl    # 3.7 MB
data/pku_saferlhf_secure/manifest.json      # Hashes for verification
```

If you **don’t** push the JSONL files, the collaborator regenerates them with:

```bash
python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
  --dataset_id PKU-Alignment/PKU-SafeRLHF \
  --dataset_config default \
  --output_dir data/pku_saferlhf_secure \
  --train_rows 10000 \
  --test_rows 1000
```

### Do NOT push

```
outputs_openllama_tonight/
outputs/
outputs_eps05/
data/pku_saferlhf_secure/raw_hf_dataset/
```

---

## 3. .gitignore Additions

Add to `.gitignore`:

```
# Outputs (LoRAs, logs)
outputs_openllama_tonight/
outputs/
outputs_eps05/

# Large data (regeneratable)
data/pku_saferlhf_secure/raw_hf_dataset/

# Build/cache
__pycache__/
*.pyc
.pytest_cache/
*.log
```

---

## 4. Dependencies

Ensure `requirements.txt` includes:

```
peft>=0.10
```

(Plus existing: torch, transformers, datasets, accelerate, etc.)

Collaborator: `conda create -n pdpo python=3.10 && conda activate pdpo && pip install -r requirements.txt`

---

## 5. RR Flip-Rate Check (Step 2A) — Data Format Note

The plan assumes `d2_rr_flipped_eps1.0_seed42.jsonl` has a `was_flipped` field. **Current `rr_stream_flip.py` does not write `was_flipped` to the main JSONL** — it only writes `prompt`, `chosen`, `rejected`.

Use the **audit file** instead:

- `outputs_openllama_tonight/preprocessing/d2_rr_audit_eps1.0_seed42.jsonl`

It has one JSON object with `observed_flip_rate`, `N`, `epsilon`, `q`, etc. The check script should read this and compare `observed_flip_rate` to expected `1/(e^ε+1) ≈ 0.269` for ε=1.0.

---

## 6. Manifest Paths (Portability)

`M2_manifest.json` uses **absolute paths** like `/users/jzhao7/PDPO/outputs_openllama_tonight/...`. A collaborator with a different layout must either:

1. Use the same directory structure, or  
2. Edit manifests to use their paths, or  
3. Load adapters from HuggingFace (after you push them) and pass HF repo IDs instead of local paths.

---

## 7. Run Order for Collaborator (From Scratch)

1. **Data:** Run `01_secure_pku_saferlhf.py` → `train_pref.jsonl`, `test_pref.jsonl`
2. **Preprocessing:** Run `run_data_prep_task.sh` → `d1_rr_flipped`, `d2_rr_flipped`, `d2_rr_audit`
3. **Stage 1:** Run `run_stage1_task.sh` for each base model
4. **Stage 2:** Run `run_stage2_mle_task.sh` and `run_stage2_sb_task.sh` for each model
5. **Evals:** Run `eval_compare.py`, `eval_preference_accuracy.py`, etc. with paths to their outputs

---

## 8. Quick Reference: Adapter Layout (Current)

```
outputs_openllama_tonight/
├── stage1/                    # One per base model
│   ├── EleutherAI--pythia-1b_stage1_rr_eps1.0_seed42/
│   ├── EleutherAI--pythia-2.8b_stage1_rr_eps1.0_seed42/
│   ├── openlm-research--open_llama_3b_v2_stage1_rr_eps1.0_seed42/
│   └── openlm-research--open_llama_7b_v2_stage1_rr_eps1.0_seed42/
├── stage2_mle_fix/            # MLE stage2 (references stage1)
├── stage2_softbayes/          # SB stage2 (references stage1)
└── preprocessing/
    ├── d1_rr_flipped_eps1.0_seed42.jsonl
    ├── d2_rr_flipped_eps1.0_seed42.jsonl
    └── d2_rr_audit_eps1.0_seed42.jsonl
```
