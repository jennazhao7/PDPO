import os, json, argparse
from pathlib import Path
from typing import List, Dict
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------- generation (deterministic, batched) ----------
@torch.no_grad()
def generate_batch(model_id: str, prompts: List[str], tokenizer, device: str = "cuda", max_new_tokens: int = 256, batch_size: int = 8) -> List[str]:
    """
    Generate responses using a model with a shared tokenizer (batched for efficiency).
    
    Args:
        model_id: Model identifier or path
        prompts: List of prompt strings
        tokenizer: Shared tokenizer to use for encoding/decoding (GPT-2-large tokenizer)
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Number of prompts to process in parallel (reduce if OOM)
    """
    print(f"Loading model: {model_id}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        # Load model - works even if model doesn't have tokenizer files
        # We use a shared tokenizer, so tokenizer files are not required
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,  # Explicit device placement
            trust_remote_code=False
        ).to(device).eval()
        print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading model {model_id}: {e}")
        print(f"   Note: Model should have config.json and model files (safetensors or bin)")
        print(f"   Tokenizer files are not required as we use a shared tokenizer")
        raise

    outs = []
    # Process in batches for efficiency
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with {model_id}"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        enc = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True,
            pad_to_multiple_of=None
        ).to(device)
        
        # Generate for batch
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)
        gen = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,            # deterministic
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,  # Use pad_token_id (set to eos_token for GPT-2)
        )
        
        # Extract only the newly generated tokens (exclude input prompt)
        input_length = input_ids.shape[1]
        generated_ids = gen[:, input_length:]
        
        # Decode only the generated part
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Store completions
        for text in texts:
            outs.append(text.strip())
        
        # Clear cache periodically to avoid memory issues
        if torch.cuda.is_available() and (i + batch_size) % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    return outs

# ---------- judge (GPT as evaluator) ----------
def get_openai_client():
    # OpenAI-compatible client (OpenAI or a compatible base_url if provided)
    from openai import OpenAI
    base = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(base_url=base) if base else OpenAI()

JUDGE_SYSTEM = (
    "You are a strict, unbiased evaluator. Compare two answers (A and B) to the same prompt. "
    "Judge helpfulness, correctness, and clarity. "
    "Reply ONLY as JSON: {\"winner\": \"A|B|Tie\", \"rationale\": \"...\"}."
)

JUDGE_USER_TMPL = """[Task]
Given one prompt and two answers (A and B), decide which answer is better.

[Prompt]
{prompt}

[Answer A]
{A}

[Answer B]
{B}

[Instructions]
- Consider accuracy, completeness, relevance, and clarity.
- If both are equally good or equally poor, choose "Tie".
- Output strict JSON only: {{"winner": "A|B|Tie", "rationale": "<brief reason>"}}
"""

def extract_json(s: str) -> Dict:
    s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            pass
    return {}

@retry(
    stop=stop_after_attempt(6), 
    wait=wait_exponential(min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def judge_once(client, model: str, prompt_text: str, A: str, B: str, temperature: float = 0.2) -> Dict:
    from openai import RateLimitError, APIError
    
    user = JUDGE_USER_TMPL.format(prompt=prompt_text, A=A, B=B)
    try:
        resp = client.chat.completions.create(
            model=model,                       # e.g., "gpt-4o-mini" or "gpt-4o"
            temperature=temperature,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user},
            ],
        )
    except RateLimitError as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            print(f"\n❌ ERROR: You have exceeded your OpenAI API quota/billing limit.")
            print(f"   Error: {error_msg}")
            print(f"   Please check your OpenAI account billing and add credits.")
            print(f"   Visit: https://platform.openai.com/account/billing")
            raise RuntimeError("OpenAI API quota exceeded. Please add credits to your account.") from e
        else:
            # Regular rate limit - retry
            print(f"⚠️  Rate limit hit, waiting and retrying...")
            raise
    
    content = resp.choices[0].message.content
    data = extract_json(content)
    w = (data.get("winner") or "").strip()
    if w not in {"A", "B", "Tie"}:
        raise ValueError(f"Bad judge output: {content}")
    return {"winner": w, "rationale": data.get("rationale", "").strip()}

def majority_vote(labels: List[str]) -> str:
    c = {"A": 0, "B": 0, "Tie": 0}
    for z in labels: c[z] = c.get(z, 0) + 1
    m = max(c.values())
    winners = [k for k, v in c.items() if v == m]
    return winners[0] if len(winners) == 1 else "Tie"

# ---------- prompt loading ----------
def load_prompts_from_truthy(n: int) -> List[str]:
    ds = load_dataset("jondurbin/truthy-dpo-v0.1")  # requires internet or cached HF dataset
    # The paper evaluated on a 100-prompt held-out slice; here we take 100 prompts deterministically.
    # Use the dataset "train" and take the last 100 unique prompts for stability.
    prompts = []
    seen = set()
    for row in ds["train"]:
        p = row["prompt"].strip()
        if p and p not in seen:
            seen.add(p); prompts.append(p)
    if len(prompts) < n:
        raise RuntimeError(f"Found only {len(prompts)} unique prompts in dataset.")
    return prompts[-n:]  # deterministic tail slice

def load_prompts_from_txt(path: str, n: int) -> List[str]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:n]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate gpt2-large DPO model against baseline")
    ap.add_argument("--model_a", default="Jennazhao7/gpt2-large-dpo-m1",
                    help="Your DPO model (A)")
    ap.add_argument("--model_b", default="Setpember/Jon_GPT2L_DPO_props_epi_point1",
                    help="Baseline PROPS DPO model (B) - epsilon=0.1")
    ap.add_argument("--n", type=int, default=100, help="Number of prompts")
    ap.add_argument("--prompts_txt", type=str, default=None,
                    help="Optional .txt with one prompt per line (overrides dataset loading)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for generation (reduce if OOM, default=4 for large model)")
    ap.add_argument("--judge_model", default="gpt-4o-mini", help="OpenAI judge model id")
    ap.add_argument("--n_votes", type=int, default=1, help="Use 1, 3, or 5 for majority vote")
    ap.add_argument("--out_csv", default="props_win_tie_results_gpt2L.csv")
    args = ap.parse_args()

    # 1) load prompts
    if args.prompts_txt:
        prompts = load_prompts_from_txt(args.prompts_txt, args.n)
    else:
        prompts = load_prompts_from_truthy(args.n)

    print(f"📝 Loaded {len(prompts)} prompts for evaluation")
    print(f"🔍 Model A (Your M1 DPO): {args.model_a}")
    print(f"🔍 Model B (Baseline PROPS): {args.model_b}")
    print(f"   Note: Models without tokenizer files will use the shared GPT-2-large tokenizer")

    # 1.5) Load shared GPT-2-large tokenizer for fair comparison
    # Note: Some models (like Setpember/Jon_GPT2L_DPO_props_epi_point1) may not have tokenizer files,
    # so we use a shared tokenizer from the base gpt2-large model
    print("\n🔧 Loading shared GPT-2-large tokenizer...")
    print("   (Using shared tokenizer for fair comparison, even if models don't have tokenizer files)")
    shared_tokenizer = AutoTokenizer.from_pretrained("gpt2-large", use_fast=True)
    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
    # Set padding side to left for decoder-only models (important for generation)
    shared_tokenizer.padding_side = "left"
    print("✅ Tokenizer loaded (padding_side='left' for decoder-only model)")

    # 2) generate from both models with identical tokenizer and decoding (batched for efficiency)
    print(f"\n🚀 Generating responses from both models (batch_size={args.batch_size})...")
    a_outs = generate_batch(args.model_a, prompts, shared_tokenizer, device=args.device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    b_outs = generate_batch(args.model_b, prompts, shared_tokenizer, device=args.device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    # 3) judge with progress saving
    client = get_openai_client()
    
    # Check if using DeepSeek and print helpful info
    if "deepseek" in args.judge_model.lower():
        base_url = os.environ.get("OPENAI_BASE_URL", "not set")
        if "deepseek" not in base_url.lower():
            print(f"⚠️  Using DeepSeek model '{args.judge_model}'")
            print(f"   Make sure OPENAI_BASE_URL is set to: https://api.deepseek.com")
            print(f"   Current OPENAI_BASE_URL: {base_url}")
    
    rows = []
    tallies = {"A": 0, "B": 0, "Tie": 0}
    
    # Try to load existing progress
    progress_file = args.out_csv.replace(".csv", "_progress.json")
    start_idx = 0
    if os.path.exists(args.out_csv):
        try:
            existing = pd.read_csv(args.out_csv)
            rows = existing.to_dict("records")
            start_idx = len(rows)
            # Recalculate tallies
            for row in rows:
                tallies[row["winner"]] += 1
            print(f"📂 Resuming from {start_idx} completed judgments...")
        except Exception as e:
            print(f"⚠️  Could not load existing progress: {e}")

    items = list(zip(prompts, a_outs, b_outs))
    if start_idx >= len(items):
        print("✅ All judgments already completed!")
    else:
        try:
            print(f"\n⚖️  Judging responses (starting from {start_idx}/{len(items)})...")
            for idx, (p, a, b) in enumerate(tqdm(items[start_idx:], desc="Judging", initial=start_idx, total=len(items))):
                ballots, rationales = [], []
                for vote_idx in range(args.n_votes):
                    try:
                        res = judge_once(client, args.judge_model, p, a, b, temperature=0.2)
                        ballots.append(res["winner"])
                        rationales.append(res["rationale"])
                        # Small delay between votes to avoid rate limits
                        if vote_idx < args.n_votes - 1:
                            time.sleep(0.5)
                    except RuntimeError as e:
                        # Quota exceeded - save progress and exit
                        if "quota exceeded" in str(e).lower():
                            print(f"\n💾 Saving progress before exiting...")
                            if rows:
                                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                                print(f"✅ Progress saved to {args.out_csv}")
                            raise
                        raise
                
                final = majority_vote(ballots)
                tallies[final] += 1
                rows.append({
                    "prompt": p,
                    "winner": final,
                    "ballots": "|".join(ballots),
                    "rationales": " || ".join(rationales),
                    "A_model": args.model_a,
                    "B_model": args.model_b,
                    "A_text": a,
                    "B_text": b
                })
                
                # Save progress every 10 items
                if (idx + 1) % 10 == 0:
                    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                
                # Small delay between prompts to avoid rate limits
                time.sleep(0.3)
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user. Saving progress...")
            if rows:
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                print(f"✅ Progress saved to {args.out_csv} ({len(rows)}/{len(items)} completed)")
            raise

    # 4) save final results and print summary
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    
    # Calculate statistics
    n = len(prompts)
    A = tallies["A"]  # Your model wins
    B = tallies["B"]  # Baseline wins
    T = tallies["Tie"]  # Ties
    
    # Win rates
    winrate_excl_ties = (A / (A+B)) if (A+B) > 0 else 0.0
    winrate_incl_ties = (A / n) if n > 0 else 0.0
    tie_rate = (T / n) if n > 0 else 0.0
    loss_rate = (B / n) if n > 0 else 0.0
    
    # Create summary CSV with interpretable results
    summary_data = {
        "Metric": [
            "Total Prompts",
            "Your Model Wins (A)",
            "Baseline Wins (B)",
            "Ties",
            "Win Rate (excluding ties)",
            "Win Rate (including ties)",
            "Tie Rate",
            "Loss Rate"
        ],
        "Count": [
            n,
            A,
            B,
            T,
            f"{winrate_excl_ties:.3f}",
            f"{winrate_incl_ties:.3f}",
            f"{tie_rate:.3f}",
            f"{loss_rate:.3f}"
        ],
        "Percentage": [
            "100.0%",
            f"{(A/n)*100:.1f}%",
            f"{(B/n)*100:.1f}%",
            f"{(T/n)*100:.1f}%",
            f"{winrate_excl_ties*100:.1f}%",
            f"{winrate_incl_ties*100:.1f}%",
            f"{tie_rate*100:.1f}%",
            f"{loss_rate*100:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = args.out_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model A (Your DPO): {args.model_a}")
    print(f"Model B (Baseline): {args.model_b}")
    print("="*60)
    print(f"Total Prompts Evaluated: {n}")
    print(f"\nResults:")
    print(f"  🏆 Your Model Wins (A): {A} ({(A/n)*100:.1f}%)")
    print(f"  📊 Baseline Wins (B):   {B} ({(B/n)*100:.1f}%)")
    print(f"  🤝 Ties:                {T} ({(T/n)*100:.1f}%)")
    print(f"\nWin Rates:")
    print(f"  Win Rate (excluding ties): {winrate_excl_ties:.3f} ({winrate_excl_ties*100:.1f}%)")
    print(f"  Win Rate (including ties): {winrate_incl_ties:.3f} ({winrate_incl_ties*100:.1f}%)")
    print(f"  Tie Rate:                  {tie_rate:.3f} ({tie_rate*100:.1f}%)")
    print(f"  Loss Rate:                 {loss_rate:.3f} ({loss_rate*100:.1f}%)")
    print("="*60)
    print(f"📄 Detailed results saved to: {args.out_csv}")
    print(f"📊 Summary saved to: {summary_csv}")
    print("="*60)

if __name__ == "__main__":
    main()

