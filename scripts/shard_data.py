import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    shard_size = (len(lines) + args.num_shards - 1) // args.num_shards

    for i in range(args.num_shards):
        shard_lines = lines[i * shard_size: (i + 1) * shard_size]
        out_path = f"{args.output_prefix}_{i}.jsonl"
        with open(out_path, "w") as out_f:
            for line in shard_lines:
                out_f.write(line + "\n")
        print(f"Wrote {len(shard_lines)} lines to {out_path}")

if __name__ == "__main__":
    main()
