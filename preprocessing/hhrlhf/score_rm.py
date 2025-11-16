import math, torch, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"   # frozen RM
MAX_LEN  = 512
BATCH    = 32   # Reduced batch size for Quadro RTX 5000
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # H100 loves bfloat16

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

# 1) Load dataset (HH-RLHF)
print("Loading HH-RLHF dataset...")
ds = load_dataset("Anthropic/hh-rlhf")

# Sample 2000 random examples for alignment data
NUM_SAMPLES = 2000
print(f"Sampling {NUM_SAMPLES} random examples from the dataset...")
if 'train' in ds:
    total_size = len(ds['train'])
    print(f"Total examples in train split: {total_size}")
    if NUM_SAMPLES > total_size:
        print(f"Warning: Requested {NUM_SAMPLES} samples but only {total_size} available. Using all {total_size} examples.")
        NUM_SAMPLES = total_size
    
    # Randomly sample NUM_SAMPLES examples
    ds = ds['train'].shuffle(seed=42).select(range(NUM_SAMPLES))
    print(f"Selected {len(ds)} examples for scoring")
else:
    print("Warning: 'train' split not found. Using available splits.")
    # If no train split, use the first available split
    split_name = list(ds.keys())[0]
    total_size = len(ds[split_name])
    print(f"Total examples in {split_name} split: {total_size}")
    if NUM_SAMPLES > total_size:
        print(f"Warning: Requested {NUM_SAMPLES} samples but only {total_size} available. Using all {total_size} examples.")
        NUM_SAMPLES = total_size
    ds = ds[split_name].shuffle(seed=42).select(range(NUM_SAMPLES))
    print(f"Selected {len(ds)} examples for scoring")

def canon(ex):
    """Canonicalize HH-RLHF format to standard prompt/chosen/rejected format."""
    # Extract prompt from chosen (both chosen and rejected should have same prompt)
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    
    ex["prompt"] = prompt
    ex["pos"] = chosen_response  # chosen response
    ex["neg"] = rejected_response  # rejected response
    return ex

print("Canonicalizing dataset format...")
ds = ds.map(canon)

# 2) Load frozen RM
print(f"Loading reward model: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
rm  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
rm.eval()
for p in rm.parameters(): p.requires_grad_(False)
print("Reward model loaded and frozen")

@torch.no_grad()
def score_batch(prompts, responses):
    texts = [f"{p}\n{r}" for p,r in zip(prompts, responses)]
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE=="cuda")):
        logits = rm(**enc).logits.squeeze(-1)  # [B]
    result = logits.float().cpu().tolist()
    # Clear GPU memory
    del enc, logits
    torch.cuda.empty_cache()
    return result

def add_margins(batch):
    rp = score_batch(batch["prompt"], batch["pos"])
    rn = score_batch(batch["prompt"], batch["neg"])
    m  = [pp - nn for pp,nn in zip(rp, rn)]
    return {"margin_raw": m, "margin_abs": [abs(x) for x in m]}

print(f"Starting scoring with batch size {BATCH}")
print(f"GPU memory before processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if DEVICE == "cuda" else "CPU mode")

ds_scored = ds.map(add_margins, batched=True, batch_size=BATCH)
ds_scored.save_to_disk("./props_with_margins_hhrlhf")
print(f"Done. Scored {len(ds_scored)} examples and saved to ./props_with_margins_hhrlhf")
if DEVICE == "cuda":
    print(f"GPU memory after processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

