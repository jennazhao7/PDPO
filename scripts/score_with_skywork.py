import argparse
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    return parser.parse_args()

def parse_hhrlhf_prompt(prompt, response):
    """
    Splits an HH-RLHF string like 'Human: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant:'
    into a list of dicts: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    messages = []
    # Split by \n\nHuman: and \n\nAssistant:
    # First, the string usually starts with 'Human: '
    if prompt.startswith("Human: "):
        prompt = prompt[7:]
    
    parts = re.split(r'\n\n(Human|Assistant): ', prompt)
    current_role = "user"
    current_content = parts[0].strip()
    
    messages.append({"role": current_role, "content": current_content})
    
    for i in range(1, len(parts), 2):
        role_marker = parts[i]
        content = parts[i+1].strip()
        role = "user" if role_marker == "Human" else "assistant"
        
        # In HH-RLHF the very last turn might be an empty Assistant prompt, which we want to append the actual response to.
        if i == len(parts) - 2 and role == "assistant" and content == "":
            content = response
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": content})
            
    # if we didn't add the response (e.g. no trailing Assistant: ), add it
    if messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": response})
        
    return messages

def main():
    args = parse_args()
    
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.input, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} lines from {args.input}")

    results = []

    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i+args.batch_size]
        
        chosen_texts = []
        rejected_texts = []
        
        for item in batch:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            chosen_msg = parse_hhrlhf_prompt(prompt, chosen)
            rejected_msg = parse_hhrlhf_prompt(prompt, rejected)
            
            try:
                c_text = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
                r_text = tokenizer.apply_chat_template(rejected_msg, tokenize=False)
            except Exception:
                c_text = prompt + " " + chosen
                r_text = prompt + " " + rejected
                
            chosen_texts.append(c_text)
            rejected_texts.append(r_text)
            
        encoded_chosen = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(model.device)
        encoded_rejected = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(model.device)
        
        with torch.no_grad():
            score_chosen = model(**encoded_chosen).logits.squeeze(-1).cpu().float().numpy()
            score_rejected = model(**encoded_rejected).logits.squeeze(-1).cpu().float().numpy()
            
        if args.batch_size == 1 or len(batch) == 1:
            score_chosen = [score_chosen.item()]
            score_rejected = [score_rejected.item()]
            
        for j, item in enumerate(batch):
            sc = float(score_chosen[j])
            sr = float(score_rejected[j])
            margin = sc - sr
            
            results.append({
                "pair_id": item.get("id", f"{i+j}"),
                "prompt": item["prompt"][:200] + "...",
                "margin_skywork": margin,
                "score_chosen": sc,
                "score_rejected": sr
            })

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Wrote {len(results)} scored pairs to {args.output}")

if __name__ == "__main__":
    main()
