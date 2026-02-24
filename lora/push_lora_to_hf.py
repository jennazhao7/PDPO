#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", required=True, help="HF repo id, e.g. Jennazhao7/pdpo-lora"
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local folder containing adapter_model.safetensors etc.",
    )
    parser.add_argument(
        "--base", required=True, help='Base model short name, e.g. "tinyllama1.1b-chat"'
    )
    parser.add_argument("--dataset", required=True, help='Dataset name, e.g. "truthydpo"')
    parser.add_argument("--stage", required=True, help='Stage name, e.g. "stage1"')
    parser.add_argument("--eps", required=True, help='Epsilon string, e.g. "1.0"')
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repo private (only used if repo is newly created)",
    )
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir

    required = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(local_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {local_dir}: {missing}\n"
            "Expected at least adapter_model.safetensors and adapter_config.json."
        )

    path_in_repo = f"{args.base}/{args.dataset}/{args.stage}/eps_{args.eps}"

    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True
        )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        path_in_repo=path_in_repo,
    )

    print(f"✅ Uploaded {local_dir} → {repo_id}/{path_in_repo}")


if __name__ == "__main__":
    main()
