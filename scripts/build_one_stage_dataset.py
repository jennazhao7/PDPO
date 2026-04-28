import argparse
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins_file", type=str, required=True)
    parser.add_argument("--original_data", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    with open(args.original_data, "r") as f:
        orig_data = [json.loads(x) for x in f if x.strip()]
        
    with open(args.margins_file, "r") as f:
        margins_data = [json.loads(x) for x in f if x.strip()]
        
    # Assume they are aligned or sort by pair_id
    if len(orig_data) != len(margins_data):
        print(f"Warning: size mismatch {len(orig_data)} vs {len(margins_data)}")
        
    margins_dict = {item["pair_id"]: item for item in margins_data}
    
    # We will compute margins std
    margins_list = [item["margin_skywork"] for item in margins_data]
    margins_std = np.std(margins_list)
    alpha = 1.0 / (margins_std + 1e-6)
    gamma = 1.0 / (1.0 + np.exp(args.epsilon))
    
    print(f"Epsilon: {args.epsilon}, Gamma: {gamma:.4f}, Alpha: {alpha:.4f}")
    
    processed = []
    
    for i, orig in enumerate(orig_data):
        pid = orig.get("id", str(i))
        if pid not in margins_dict:
            # fallback
            pid = str(i)
            
        m_item = margins_dict[pid]
        m_skywork = m_item["margin_skywork"]
        
        # original label is always 1 (chosen is chosen)
        # RR applied:
        flip = np.random.rand() < gamma
        rr_label = 0 if flip else 1
        
        if not flip:
            signed_margin = m_skywork
        else:
            signed_margin = -m_skywork
            
        p_S = sigmoid(alpha * signed_margin)
        
        # q_i
        q_i = (p_S * (1 - gamma)) / (p_S * (1 - gamma) + (1 - p_S) * gamma)
        
        new_item = {
            "pair_id": pid,
            "prompt": orig["prompt"],
            "chosen": orig["chosen"],
            "rejected": orig["rejected"],
            "rr_label": rr_label,
            "true_label": 1,
            "margin_skywork": m_skywork,
            "q_i": q_i,
            "w_i": q_i,
            "kept": True
        }
        processed.append(new_item)
        
    # Output arrays
    q_values = np.array([p["q_i"] for p in processed])
    print(f"N total pairs: {len(q_values)}")
    print(f"q distribution: mean={q_values.mean():.3f}, std={q_values.std():.3f}")
    print(f"q quantiles: 10%={np.percentile(q_values, 10):.3f}, 50%={np.percentile(q_values, 50):.3f}, 90%={np.percentile(q_values, 90):.3f}")
    print(f"Fraction q > 0.9: {(q_values > 0.9).mean():.3f}")
    print(f"Fraction q < 0.1: {(q_values < 0.1).mean():.3f}")
    print(f"Fraction q in [0.4, 0.6] (ambiguous): {((q_values > 0.4) & (q_values < 0.6)).mean():.3f}")
    
    confidence = np.abs(q_values - 0.5)
    
    def write_subset(keep_frac, name):
        threshold = np.quantile(confidence, 1 - keep_frac)
        keep_mask = confidence >= threshold
        keep_count = keep_mask.sum()
        print(f"After selection (keep={keep_frac}): {keep_count} pairs kept")
        
        out_path = f"{args.output_prefix}_keep{keep_frac}.jsonl"
        with open(out_path, "w") as f:
            for item, keep in zip(processed, keep_mask):
                if keep:
                    # If rr_label is 0, we flip chosen and rejected for the trainer
                    # Wait! DPO trainer expects 'chosen' and 'rejected' to be the preferred and dispreferred.
                    # If rr_label == 0, the preferred is the original 'rejected'.
                    final_item = dict(item)
                    if item["rr_label"] == 0:
                        final_item["chosen"] = item["rejected"]
                        final_item["rejected"] = item["chosen"]
                    f.write(json.dumps(final_item) + "\n")
        print(f"Wrote to {out_path}")
        
    write_subset(0.5, "0.5")
    write_subset(1.0, "1.0")
    write_subset(0.25, "0.25")
    
    # RR only control:
    out_rr_only = f"{args.output_prefix}_rr_only.jsonl"
    with open(out_rr_only, "w") as f:
        for item in processed:
            final_item = dict(item)
            final_item["w_i"] = 1.0 # no weighting
            if item["rr_label"] == 0:
                final_item["chosen"] = item["rejected"]
                final_item["rejected"] = item["chosen"]
            f.write(json.dumps(final_item) + "\n")
    print(f"Wrote control to {out_rr_only}")

if __name__ == "__main__":
    main()
