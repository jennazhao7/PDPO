import math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"   # frozen RM
MAX_LEN  = 512
BATCH    = 8   # Small batch for testing
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load small subset of dataset
print("\nLoading dataset...")
ds = load_dataset("jondurbin/truthy-dpo-v0.1")
print(f"Dataset loaded. Total examples: {len(ds['train'])}")

# Take a small subset for testing (first 5 examples)
test_ds = ds['train'].select(range(5))
print(f"Using subset of {len(test_ds)} examples")

def canon(ex):
    ex["prompt"] = ex.get("prompt") or ex.get("instruction")
    ex["pos"]    = ex.get("chosen") or ex.get("response_chosen") or ex.get("output_1") or ex.get("response_0")
    ex["neg"]    = ex.get("rejected") or ex.get("response_rejected") or ex.get("output_2") or ex.get("response_1")
    return ex

test_ds = test_ds.map(canon)
print("Dataset standardized")

# Load frozen RM
print(f"\nLoading RM: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
rm  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
rm.eval()
for p in rm.parameters(): p.requires_grad_(False)
print("RM loaded and frozen")

@torch.no_grad()
def score_batch(prompts, responses):
    texts = [f"{p}\n{r}" for p,r in zip(prompts, responses)]
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE=="cuda")):
        logits = rm(**enc).logits.squeeze(-1)  # [B]
    return logits.float().cpu().tolist()

# Score the test subset
print(f"\nScoring {len(test_ds)} examples...")
print("=" * 80)

for i, example in enumerate(test_ds):
    print(f"\nExample {i+1}:")
    print(f"Prompt: {example['prompt'][:100]}...")
    
    # Score chosen response
    chosen_score = score_batch([example['prompt']], [example['pos']])[0]
    print(f"Chosen response: {example['pos'][:100]}...")
    print(f"Chosen score: {chosen_score:.4f}")
    
    # Score rejected response  
    rejected_score = score_batch([example['prompt']], [example['neg']])[0]
    print(f"Rejected response: {example['neg'][:100]}...")
    print(f"Rejected score: {rejected_score:.4f}")
    
    # Calculate margin
    margin = chosen_score - rejected_score
    print(f"Margin (chosen - rejected): {margin:.4f}")
    print("-" * 80)

print(f"\nTest completed successfully on {DEVICE}!")
print("All examples processed and scored.")
