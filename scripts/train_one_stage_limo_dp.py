import os
import gc
import json
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

def load_causal_lm_compat(model_path, **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

class OneStagePairDataset(TorchDataset):
    def __init__(self, data, tok, max_len):
        self.data = data
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_ids = self.tok.encode(item["prompt"], add_special_tokens=True)
        chosen_ids = self.tok.encode(item["chosen"], add_special_tokens=False)
        rejected_ids = self.tok.encode(item["rejected"], add_special_tokens=False)
        return {
            "chosen_ids": (prompt_ids + chosen_ids)[: self.max_len],
            "rejected_ids": (prompt_ids + rejected_ids)[: self.max_len],
            "prompt_len": len(prompt_ids),
            "w": item["w_i"]
        }

def pad_to_max(sequences, pad_id):
    max_len = max(len(s) for s in sequences)
    return torch.tensor([s + [pad_id] * (max_len - len(s)) for s in sequences], dtype=torch.long)

def collate_fn(batch, pad_id):
    return {
        "chosen_ids": pad_to_max([b["chosen_ids"] for b in batch], pad_id),
        "rejected_ids": pad_to_max([b["rejected_ids"] for b in batch], pad_id),
        "prompt_lens": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "w": torch.tensor([b["w"] for b in batch], dtype=torch.float32)
    }

def compute_response_logprobs(model, input_ids, prompt_lens, pad_id):
    outputs = model(input_ids=input_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    B, L_minus_1 = token_log_probs.shape
    positions = torch.arange(L_minus_1, device=input_ids.device).unsqueeze(0).expand(B, -1)
    response_mask = positions >= (prompt_lens.unsqueeze(1) - 1)
    pad_mask = shift_labels != pad_id
    mask = response_mask & pad_mask

    return (token_log_probs * mask.float()).sum(dim=1)

def soft_bayes_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, w, beta):
    m = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    loss = -(w * F.logsigmoid(beta * m) + (1.0 - w) * F.logsigmoid(-beta * m))
    return loss.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_weights", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--ga", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np_seed = args.seed
    import numpy as np
    np.random.seed(np_seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    with open(args.train_data, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} items from {args.train_data}")

    use_fast = False if "open_llama" in args.base_model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dtype = torch.bfloat16 if (cuda_ok and args.bf16) else (torch.float16 if cuda_ok else torch.float32)

    pair_ds = OneStagePairDataset(data, tok, args.max_len)
    train_dl = DataLoader(pair_ds, batch_size=args.bsz, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))

    print(f"Loading {args.base_model}...")
    base_model = load_causal_lm_compat(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    stage2_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    m2 = get_peft_model(base_model, stage2_config, adapter_name="stage2")
    m2 = m2.to(device).train()

    optimizer = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=args.lr, eps=1e-6)
    
    total_steps = (len(train_dl) // args.ga) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    global_step = 0
    accum_loss = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        for step_idx, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            w = batch["w"].to(device)
            if not args.use_weights:
                w = torch.ones_like(w)

            with m2.disable_adapter():
                with torch.no_grad():
                    ref_c = compute_response_logprobs(m2, chosen_ids, prompt_lens, pad_id)
                    ref_r = compute_response_logprobs(m2, rejected_ids, prompt_lens, pad_id)
            
            pi_c = compute_response_logprobs(m2, chosen_ids, prompt_lens, pad_id)
            pi_r = compute_response_logprobs(m2, rejected_ids, prompt_lens, pad_id)

            loss = soft_bayes_loss(pi_c, pi_r, ref_c, ref_r, w, args.beta)
            loss = loss / args.ga
            loss.backward()

            accum_loss += loss.item()

            if (step_idx + 1) % args.ga == 0 or (step_idx + 1) == len(train_dl):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_loss = 0.0
                
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    m2.save_pretrained(args.output_dir)
    
    manifest = {
        "adapter_path": args.output_dir,
        "base_model": args.base_model,
        "method": "onestage_limo_dp"
    }
    with open(os.path.join(args.output_dir, "M2_manifest.json"), "w") as f:
        json.dump(manifest, f)
        
    print(f"Saved M2_manifest.json to {args.output_dir}")

if __name__ == "__main__":
    main()
