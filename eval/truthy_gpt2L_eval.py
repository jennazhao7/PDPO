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

# ---------- generation (deterministic) ----------
@torch.no_grad()
def generate_batch(model_id: str, prompts: List[str], device: str = "cuda", max_new_tokens: int = 256) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.eos_token_id is None and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or ""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    outs = []
    for p in tqdm(prompts, desc=f"Generating with {model_id}"):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(device)
        gen = mdl.generate(
            **enc,
            do_sample=False,            # deterministic
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id
        )
        text = tok.decode(gen[0], skip_special_tokens=True)
        # keep only the completion portion for fairness
        comp = text[len(p):].strip() if text.startswith(p) else text
        outs.append(comp)
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
            model=model,                       # e.g., "gpt-4.1-mini" or "gpt-4o"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", default="Jennazhao7/gpt2-large-dpo-m1",
                    help="Your model (A)")
    ap.add_argument("--model_b", default="Setpember/Jon_GPT2L_DPO_epi_point1",
                    help="Baseline model (B) from PROPs collection")
    ap.add_argument("--n", type=int, default=100, help="Number of prompts")
    ap.add_argument("--prompts_txt", type=str, default=None,
                    help="Optional .txt with one prompt per line (overrides dataset loading)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--judge_model", default="gpt-4o-mini", help="OpenAI judge model id")
    ap.add_argument("--n_votes", type=int, default=1, help="Use 1, 3, or 5 for majority vote")
    ap.add_argument("--out_csv", default="props_win_tie_results.csv")
    args = ap.parse_args()

    # 1) load prompts
    if args.prompts_txt:
        prompts = load_prompts_from_txt(args.prompts_txt, args.n)
    else:
        prompts = load_prompts_from_truthy(args.n)

    # 2) generate from both models with identical decoding
    a_outs = generate_batch(args.model_a, prompts, device=args.device, max_new_tokens=args.max_new_tokens)
    b_outs = generate_batch(args.model_b, prompts, device=args.device, max_new_tokens=args.max_new_tokens)

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
    n = len(prompts); A=tallies["A"]; B=tallies["B"]; T=tallies["Tie"]
    winrate_excl_ties = (A / (A+B)) if (A+B) > 0 else 0.0
    print("\n=== Summary ===")
    print(f"Total prompts: {n}")
    print(f"A wins: {A}, B wins: {B}, Ties: {T}")
    print(f"Win rate (A vs B, drop ties): {winrate_excl_ties:.3f}")
    print(f"CSV saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
