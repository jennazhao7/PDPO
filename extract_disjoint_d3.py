import sys
import hashlib
import json
import os
from typing import Dict, Iterable, Optional
from datasets import load_dataset


def canonical_example_id(prompt: str, r0: str, r1: str) -> str:
    raw = f"{prompt}\n<<R0>>\n{r0}\n<<R1>>\n{r1}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]

def normalize_row(row: Dict) -> Optional[Dict]:
    prompt = str(row.get("prompt", "")).strip()
    r0 = str(row.get("response_0", "")).strip()
    r1 = str(row.get("response_1", "")).strip()
    better = row.get("better_response_id", None)
    safer = row.get("safer_response_id", None)

    if not prompt or not r0 or not r1:
        return None
    if better not in (0, 1):
        return None

    chosen = r0 if better == 0 else r1
    rejected = r1 if better == 0 else r0

    return {
        "id": canonical_example_id(prompt, r0, r1),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "response_0": r0,
        "response_1": r1,
        "better_response_id": int(better),
        "safer_response_id": int(safer) if safer in (0, 1) else None,
        "response_0_source": row.get("response_0_source"),
        "response_1_source": row.get("response_1_source"),
        "prompt_source": row.get("prompt_source"),
    }

def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    print("Loading PKU-SafeRLHF from HuggingFace...")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    
    start_idx = 10000
    end_idx = 16000
    
    new_ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
    normalized = []
    for row in new_ds:
        item = normalize_row(row)
        if item is not None:
            normalized.append(item)
            if len(normalized) == 5000:
                break
                
    out_path = "data/pku_saferlhf_secure/d3_train_pref.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_jsonl(out_path, normalized)
    print(f"Saved {len(normalized)} rows for completely disjoint D3 to {out_path}.")

if __name__ == "__main__":
    main()
