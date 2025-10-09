import math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"   # frozen RM
MAX_LEN  = 512
BATCH    = 128   # H100 can usually handle 128 for 512 tokens
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # H100 loves bfloat16

# 1) Load dataset (example: Truthy-DPO)
ds = load_dataset("jondurbin/truthy-dpo-v0.1")

def canon(ex):
    ex["prompt"] = ex.get("prompt") or ex.get("instruction")
    ex["pos"]    = ex.get("chosen") or ex.get("response_chosen") or ex.get("output_1") or ex.get("response_0")
    ex["neg"]    = ex.get("rejected") or ex.get("response_rejected") or ex.get("output_2") or ex.get("response_1")
    return ex
ds = ds.map(canon)

# 2) Load frozen RM
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
rm  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
rm.eval()
for p in rm.parameters(): p.requires_grad_(False)

@torch.no_grad()
def score_batch(prompts, responses):
    texts = [f"{p}\n{r}" for p,r in zip(prompts, responses)]
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE=="cuda")):
        logits = rm(**enc).logits.squeeze(-1)  # [B]
    return logits.float().cpu().tolist()

def add_margins(batch):
    rp = score_batch(batch["prompt"], batch["pos"])
    rn = score_batch(batch["prompt"], batch["neg"])
    m  = [pp - nn for pp,nn in zip(rp, rn)]
    return {"margin_raw": m, "margin_abs": [abs(x) for x in m]}

ds_scored = ds.map(add_margins, batched=True, batch_size=BATCH)
ds_scored.save_to_disk("./props_with_margins_truthy")
print("Done. Saved to ./props_with_margins_truthy")
