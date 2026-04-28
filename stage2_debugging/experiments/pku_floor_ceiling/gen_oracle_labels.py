#!/usr/bin/env python3
"""
Attach was_flipped to each D2 row by re-running the deterministic RR flip
with (id, epsilon, seed) — identical to what rr_stream_flip produced.

Output JSONL: same rows as D2 but with `was_flipped` field added.

Usage:
  python gen_oracle_labels.py \
    --data  ../../preprocessing/d2_rr_flipped_pku_eps1.0_seed42.jsonl \
    --out   d2_pku_with_flipped_eps1.0_seed42.jsonl \
    --epsilon 1.0 --seed 42
"""
import argparse, hashlib, json, math, sys

def q_from_epsilon(epsilon: float) -> float:
    return 1.0 / (math.exp(epsilon) + 1.0)

def stable_uniform(seed: int, example_id: str) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    h.update(b"|")
    h.update(str(example_id).encode())
    digest = h.digest()
    return int.from_bytes(digest, byteorder="big") / 2**64

def get_example_id(row: dict, id_key: str = "id") -> str:
    if id_key in row and row[id_key] is not None:
        return str(row[id_key])
    raw = f"{row.get('prompt','')}\n{row.get('chosen','')}\n{row.get('rejected','')}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    q = q_from_epsilon(args.epsilon)
    print(f"epsilon={args.epsilon}  q={q:.4f}  (expected flip rate ~{q:.1%})")

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    n_flipped = 0
    with open(args.out, "w") as fout:
        for row in rows:
            eid = get_example_id(row)
            u   = stable_uniform(args.seed, eid)
            was_flipped = u < q
            if was_flipped:
                n_flipped += 1
            out_row = dict(row)
            out_row["was_flipped"] = was_flipped
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    N = len(rows)
    print(f"Wrote {N} rows → {args.out}")
    print(f"Observed flip rate: {n_flipped}/{N} = {n_flipped/N:.3f}  (expected {q:.3f})")
    assert abs(n_flipped/N - q) < 0.02, "Flip rate mismatch — check epsilon/seed!"
    print("✅ Flip rate sanity check passed.")

if __name__ == "__main__":
    sys.exit(main())
