import json

true_flips = 0
total = 0

with open("outputs_new_models/preprocessing/d1_rr_audit_eps1.0_seed42.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if "observed_flip_rate" in data:
            print(f"TruthyDPO Audit Log Empirical Rate: {data['observed_flip_rate']:.4f}")
            print(f"TruthyDPO Audit Log Expected Rate (q): {data['q']:.4f}")
            if abs(data['observed_flip_rate'] - data['q']) < 0.02:
                print("✅ PASS: Matches ε=1.0 theoretically expected rate based on audit log")
            else:
                print("❌ FAIL: Does not match theoretical rate in audit log")

