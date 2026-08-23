#!/usr/bin/env python3
"""Matched fresh-LoRA trainer using precomputed base reference logprobs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from stage2_debugging.matched_core import (
        MATCHED_DEFAULTS,
        deterministic_subset_indices,
        dp_sgd_epsilon_rdp,
        estimate_eta_em,
        load_reference_cache,
        penalized_margin_term,
        read_jsonl,
        redpo_loss,
        relative_margin_clip_values,
        robust_channel_losses,
        true_eta,
        write_jsonl,
        write_tiny_artifacts,
    )
    from stage2_debugging.ref_logprob_core import (
        LOGPROB_DTYPE,
        POLICY_DTYPE,
        REFERENCE_DTYPE,
        TRUNCATION_MODE,
        encode_prompt_response,
        torch_padded_response_logprobs,
    )
except ModuleNotFoundError:
    from matched_core import (  # type: ignore
        MATCHED_DEFAULTS,
        deterministic_subset_indices,
        dp_sgd_epsilon_rdp,
        estimate_eta_em,
        load_reference_cache,
        penalized_margin_term,
        read_jsonl,
        redpo_loss,
        relative_margin_clip_values,
        robust_channel_losses,
        true_eta,
        write_jsonl,
        write_tiny_artifacts,
    )
    from ref_logprob_core import (  # type: ignore
        LOGPROB_DTYPE,
        POLICY_DTYPE,
        REFERENCE_DTYPE,
        TRUNCATION_MODE,
        encode_prompt_response,
        torch_padded_response_logprobs,
    )


CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def file_sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_version_of(data_path) -> str | None:
    """Split generation from the canonical directory naming (…_secure_v3/… -> "v3")."""
    import re

    match = re.search(r"_secure(?:_(v\d+))?/", str(data_path))
    if not match:
        return None
    return match.group(1) or "v1"


def total_optimizer_steps(n_batches: int, ga: int, epochs: int, max_steps: int) -> int:
    """Optimizer steps the training loop will take.

    The loop steps every ``ga`` batches and also on the final batch of each epoch, so an epoch
    contributes ceil(n_batches / ga) steps.
    """
    per_epoch = math.ceil(n_batches / ga) if n_batches else 0
    total = per_epoch * epochs
    if max_steps > 0:
        total = min(total, max_steps)
    return total


def checkpoint_boundaries(
    total_steps: int, fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS
) -> dict[int, int]:
    """Map each checkpoint percentage to the optimizer step at which it is taken.

    Clamped into [1, total_steps] so short runs still emit every checkpoint. Percentages can
    share a step when total_steps < 4; each percentage is still emitted exactly once.
    """
    if total_steps <= 0:
        return {}
    return {
        int(round(fraction * 100)): max(1, min(total_steps, int(round(fraction * total_steps))))
        for fraction in fractions
    }


def build_parser(default_method: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["rdpo", "redpo", "random_subset", "mle", "penalized_margin", "margin_control"],
        default=default_method,
    )
    parser.add_argument("--model", required=False, default="tiny-fixture")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ref_logps", default=None)
    parser.add_argument("--expect_cache_eps", type=float, default=None,
                        help="Arm the reference cache must declare. Defaults to --epsilon; set "
                             "explicitly when --epsilon is a loss parameter rather than the RR arm "
                             "(Stage 2 passes eps=1.0 to the MLE loss at rr_eps=0.0).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest_out", default=None)
    parser.add_argument("--epsilon", "--rr_epsilon", dest="epsilon", type=float, default=1.0)
    parser.add_argument("--eta_fixed", type=float, default=None)
    parser.add_argument("--eta_initial", type=float, default=0.75)
    parser.add_argument("--eta_em_steps", type=int, default=50)
    parser.add_argument("--keep_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=MATCHED_DEFAULTS["beta"])
    parser.add_argument("--lr", type=float, default=MATCHED_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=MATCHED_DEFAULTS["epochs"])
    parser.add_argument("--max_steps", type=int, default=MATCHED_DEFAULTS["max_steps"])
    parser.add_argument("--bsz", type=int, default=MATCHED_DEFAULTS["bsz"])
    parser.add_argument("--ga", type=int, default=MATCHED_DEFAULTS["ga"])
    parser.add_argument("--max_len", type=int, default=MATCHED_DEFAULTS["max_len"])
    parser.add_argument("--lora_r", type=int, default=MATCHED_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=MATCHED_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=MATCHED_DEFAULTS["lora_dropout"])
    parser.add_argument("--target_modules", default=MATCHED_DEFAULTS["target_modules"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tiny_test", action="store_true")
    parser.add_argument(
        "--penalty_variant",
        choices=["none", "global", "selective"],
        default="none",
        help="Margin-control penalty variant for --method penalized_margin.",
    )
    parser.add_argument("--penalty_lambda", type=float, default=0.0)
    parser.add_argument(
        "--penalty_tau",
        type=float,
        default=0.0,
        help="Post-beta margin threshold for the selective penalty.",
    )
    parser.add_argument(
        "--control_mechanism",
        choices=["none", "per_example_grad_clip", "relative_margin_clip", "dp_sgd"],
        default="none",
        help="Influence-limiting mechanism for --method margin_control.",
    )
    parser.add_argument(
        "--control_value",
        type=float,
        default=-1.0,
        help="C for per-example gradient clipping, or cap for relative-margin clipping. Negative means no-op.",
    )
    parser.add_argument(
        "--median_window",
        type=int,
        default=128,
        help="Rolling detached-margin window for relative-margin clipping.",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=0.0,
        help="Gaussian noise multiplier for --control_mechanism dp_sgd.",
    )
    parser.add_argument(
        "--dp_delta",
        type=float,
        default=1e-5,
        help="Delta for gradient-level DP-SGD accounting.",
    )
    return parser


def _tiny_logits(rows: list[dict[str, Any]]) -> list[float]:
    return [(len(str(row["chosen"])) - len(str(row["rejected"]))) / 10.0 for row in rows]


def run_tiny(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    method = str(args.method)
    selected = rows
    if method == "random_subset":
        selected = [rows[index] for index in deterministic_subset_indices(len(rows), args.keep_fraction, args.seed)]
    logits = _tiny_logits(selected)
    extra: dict[str, Any] = {"epsilon": args.epsilon, "selected_n": len(selected)}
    if method == "rdpo":
        eta = true_eta(args.epsilon)
        extra.update({"eta_true": eta, "loss": sum(robust_channel_losses(logits, eta)) / len(logits)})
    elif method == "redpo":
        eta_true_value = true_eta(args.epsilon)
        if args.eta_fixed is not None:
            eta, responsibilities, iterations = args.eta_fixed, [], 0
        else:
            eta, responsibilities, iterations = estimate_eta_em(
                logits, initial_eta=args.eta_initial, max_iter=args.eta_em_steps
            )
        extra.update(
            {
                "eta_estimated": eta,
                "eta_true": eta_true_value,
                "eta_em_iterations": iterations,
                "mean_reliability_posterior": (
                    sum(responsibilities) / len(responsibilities) if responsibilities else eta
                ),
                "loss": redpo_loss(logits, eta),
            }
        )
    elif method in {"random_subset", "mle", "penalized_margin", "margin_control"}:
        losses = [math.log1p(math.exp(-value)) for value in logits]
        penalty = args.penalty_lambda * penalized_margin_term(
            logits, args.penalty_variant, args.penalty_tau
        )
        effective_logits = logits
        if method == "margin_control" and args.control_mechanism == "relative_margin_clip":
            effective_logits = relative_margin_clip_values(logits, median=0.0, cap=args.control_value)
            losses = [math.log1p(math.exp(-value)) for value in effective_logits]
        extra.update(
            {
                "loss": sum(losses) / len(losses) + penalty,
                "dpo_loss": sum(losses) / len(losses),
                "margin_penalty": penalty,
                "penalty_variant": args.penalty_variant,
                "penalty_lambda": args.penalty_lambda,
                "penalty_tau": args.penalty_tau,
                "control_mechanism": args.control_mechanism if method == "margin_control" else None,
                "control_value": args.control_value if method == "margin_control" else None,
            }
        )
    write_tiny_artifacts(args.out, method, selected, extra)
    return 0


def train_real(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    """Lazy-imported real trainer; never constructs a reference model."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.lora_r != 16:
        raise ValueError("matched comparison requires --lora_r 16")
    if not args.ref_logps:
        raise ValueError("--ref_logps is required; reference-model loading is forbidden")
    if args.method == "penalized_margin":
        if args.penalty_lambda < 0:
            raise ValueError("--penalty_lambda must be nonnegative")
        if args.penalty_tau < 0:
            raise ValueError("--penalty_tau must be nonnegative")
    elif args.penalty_lambda or args.penalty_variant != "none" or args.penalty_tau:
        raise ValueError("margin-penalty arguments are only valid for --method penalized_margin")
    if args.method == "margin_control":
        if args.control_mechanism == "none" and args.control_value >= 0:
            raise ValueError("--control_value requires an active --control_mechanism")
        if args.control_mechanism != "none" and args.median_window <= 0:
            raise ValueError("--median_window must be positive")
        if args.control_mechanism == "dp_sgd":
            if args.control_value <= 0:
                raise ValueError("--control_value must be positive for --control_mechanism dp_sgd")
            if args.noise_multiplier < 0:
                raise ValueError("--noise_multiplier must be nonnegative")
            if not (0.0 < args.dp_delta < 1.0):
                raise ValueError("--dp_delta must be in (0, 1)")
        elif args.noise_multiplier:
            raise ValueError("--noise_multiplier requires --control_mechanism dp_sgd")
    elif args.control_mechanism != "none" or args.control_value >= 0:
        raise ValueError("influence-control arguments are only valid for --method margin_control")
    elif args.noise_multiplier:
        raise ValueError("--noise_multiplier is only valid for --method margin_control")

    # expected_rr_eps asserts the cache was built for THIS arm. The RR swap changes pair_key so
    # a wrong-arm cache was already refused, but only via a per-row lookup miss; this states it.
    ref_chosen, ref_rejected = load_reference_cache(
        args.ref_logps,
        rows,
        args.model,
        expected_scoring_dtype=REFERENCE_DTYPE,
        expected_truncation=TRUNCATION_MODE,
        expected_max_len=args.max_len,
        expected_rr_eps=args.epsilon if args.expect_cache_eps is None else args.expect_cache_eps,
        expected_cache_role="train_reference_arm_orientation",
    )
    selected_member_rows: list[dict[str, Any]] | None = None
    if args.method == "random_subset":
        indices = deterministic_subset_indices(len(rows), args.keep_fraction, args.seed)
        rows = [rows[i] for i in indices]
        selected_member_rows = list(rows)
        ref_chosen = [ref_chosen[i] for i in indices]
        ref_rejected = [ref_rejected[i] for i in indices]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast="open_llama" not in args.model.lower())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id

    class PairDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            chosen_ids, chosen_prompt_len = encode_prompt_response(
                tokenizer, str(row["prompt"]), str(row["chosen"]), args.max_len
            )
            rejected_ids, rejected_prompt_len = encode_prompt_response(
                tokenizer, str(row["prompt"]), str(row["rejected"]), args.max_len
            )
            return {
                "chosen": chosen_ids,
                "rejected": rejected_ids,
                "chosen_prompt_len": chosen_prompt_len,
                "rejected_prompt_len": rejected_prompt_len,
                "ref_c": ref_chosen[index],
                "ref_r": ref_rejected[index],
            }

    def pad(values):
        width = max(len(value) for value in values)
        tokens = torch.full((len(values), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(values), width), dtype=torch.long)
        for index, value in enumerate(values):
            tokens[index, :len(value)] = torch.tensor(value, dtype=torch.long)
            attention[index, :len(value)] = 1
        return tokens, attention

    def collate(batch):
        chosen, chosen_attention = pad([item["chosen"] for item in batch])
        rejected, rejected_attention = pad([item["rejected"] for item in batch])
        return {
            "chosen": chosen,
            "chosen_attention": chosen_attention,
            "rejected": rejected,
            "rejected_attention": rejected_attention,
            "chosen_prompt_len": torch.tensor([item["chosen_prompt_len"] for item in batch]),
            "rejected_prompt_len": torch.tensor([item["rejected_prompt_len"] for item in batch]),
            "ref_c": torch.tensor([item["ref_c"] for item in batch], dtype=torch.float32),
            "ref_r": torch.tensor([item["ref_r"] for item in batch], dtype=torch.float32),
        }

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if args.fp32 and args.bf16:
        raise ValueError("--fp32 and --bf16 are mutually exclusive")
    dtype = (
        torch.float32
        if args.fp32 or not cuda
        else (torch.bfloat16 if args.bf16 else torch.float16)
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=False
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    config = LoraConfig(
        r=16,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, config, adapter_name="stage2").to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        eps=1e-6,
    )
    loader = DataLoader(PairDataset(), batch_size=args.bsz, shuffle=True, collate_fn=collate)
    optimizer.zero_grad()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accumulated_grads: list[torch.Tensor] | None = None
    accumulated_examples = 0
    step = 0
    eta_estimated = args.eta_fixed
    last_loss = 0.0
    margin_history: list[float] = []

    # Intermediate checkpoints: one adapter per quarter of training, each with the loss observed
    # at that point, so the operating-point sweep can evaluate partial fits rather than only the
    # final model.
    planned_total_steps = total_optimizer_steps(
        len(loader), args.ga, args.epochs, args.max_steps
    )
    pending_checkpoints = checkpoint_boundaries(planned_total_steps)
    checkpoints_written: list[dict[str, Any]] = []

    def write_checkpoint(percent: int, at_step: int) -> None:
        directory = Path(args.out) / f"checkpoint_p{percent:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory, adapter_name="stage2")
        record = {
            "percent": percent,
            "step": at_step,
            "planned_total_steps": planned_total_steps,
            "loss": last_loss,
            "path": str(directory),
        }
        (directory / "checkpoint_state.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Each checkpoint needs its own M2_manifest.json: the evaluators load a model through a
        # manifest, not through an adapter directory, so a checkpoint without one cannot be
        # evaluated even though its weights are present.
        (directory / "M2_manifest.json").write_text(
            json.dumps(
                {
                    "base_model": args.model,
                    "adapters": [{"name": "stage2", "path": str(directory.resolve())}],
                    "method": args.method,
                    "params": {
                        "fresh_from_base": True,
                        "reference": "cached_base_logprobs",
                        # Provenance. The seed drives LoRA init, shuffling and the MIA subsample;
                        # recording it only in the output directory name is not provenance, since
                        # a rename loses it.
                        "seed": args.seed,
                        "n_train": len(rows),
                        "data_jsonl": str(args.data),
                        "data_sha256": file_sha256(args.data),
                        "split_version": split_version_of(args.data),
                        # No git commit can identify this run: the source bundle is built from an
                        # uncommitted tree. The immutable tarball is the durable code identifier.
                        "source_bundle": os.environ.get("PDPO_SOURCE_BUNDLE"),
                        "source_bundle_sha256": os.environ.get("PDPO_SOURCE_BUNDLE_SHA256"),
                        "precision": {
                            "policy_weights": (
                                POLICY_DTYPE
                                if args.fp32
                                else ("bfloat16" if args.bf16 else "float16")
                            ),
                            "policy_logprobs": LOGPROB_DTYPE,
                            "reference_cache": REFERENCE_DTYPE,
                            "margin": LOGPROB_DTYPE,
                        },
                        "truncation": TRUNCATION_MODE,
                        "max_len": args.max_len,
                        "lora_rank": args.lora_r,
                        "epochs": args.epochs,
                        "beta": args.beta,
                        "lr": args.lr,
                        "checkpoint": record,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoints_written.append(record)

    def current_margin_median(batch_margin: torch.Tensor) -> torch.Tensor:
        if margin_history:
            history = torch.tensor(margin_history, dtype=batch_margin.dtype, device=batch_margin.device)
            return torch.median(history)
        return torch.median(batch_margin.detach())

    def update_margin_history(batch_margin: torch.Tensor) -> None:
        if args.method == "margin_control" and args.control_mechanism == "relative_margin_clip":
            margin_history.extend(float(value) for value in batch_margin.detach().float().cpu().tolist())
            if len(margin_history) > args.median_window:
                del margin_history[:-args.median_window]

    def accumulate_clipped_microbatch(loss_value: torch.Tensor) -> None:
        nonlocal accumulated_grads, accumulated_examples
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        if args.control_value > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.control_value)
        if accumulated_grads is None:
            accumulated_grads = [
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                for parameter in trainable_parameters
            ]
        for index, parameter in enumerate(trainable_parameters):
            if parameter.grad is not None:
                accumulated_grads[index].add_(parameter.grad.detach(), alpha=1.0 / args.ga)
        accumulated_examples += 1
        optimizer.zero_grad(set_to_none=True)

    def flush_accumulated_grads() -> None:
        nonlocal accumulated_grads, accumulated_examples
        if accumulated_grads is None:
            return
        optimizer.zero_grad(set_to_none=True)
        for parameter, grad in zip(trainable_parameters, accumulated_grads):
            noisy_grad = grad.clone()
            if args.control_mechanism == "dp_sgd" and args.noise_multiplier > 0:
                # accumulated_grads stores the averaged clipped gradient over args.ga
                # microbatches, so add Gaussian noise at the same averaged scale.
                noise_std = args.noise_multiplier * args.control_value / max(1, accumulated_examples)
                noisy_grad.add_(torch.randn_like(noisy_grad), alpha=noise_std)
            parameter.grad = noisy_grad
        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulated_grads = None
        accumulated_examples = 0

    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            chosen_lp = torch_padded_response_logprobs(
                model,
                batch["chosen"].to(device),
                batch["chosen_attention"].to(device),
                batch["chosen_prompt_len"],
            )
            rejected_lp = torch_padded_response_logprobs(
                model,
                batch["rejected"].to(device),
                batch["rejected_attention"].to(device),
                batch["rejected_prompt_len"],
            )
            ref_c = batch["ref_c"].to(device)
            ref_r = batch["ref_r"].to(device)
            margin = args.beta * (
                (chosen_lp - ref_c)
                - (rejected_lp - ref_r)
            )
            if args.method == "rdpo":
                eta = true_eta(args.epsilon)
                gamma = 1.0 - eta
                loss = (
                    eta * F.softplus(-margin) - gamma * F.softplus(margin)
                ).mean() / (2.0 * eta - 1.0)
            elif args.method == "redpo":
                if args.eta_fixed is None:
                    eta_estimated, _, _ = estimate_eta_em(
                        margin.detach().float().cpu().tolist(),
                        initial_eta=(
                            eta_estimated if eta_estimated is not None else args.eta_initial
                        ),
                        max_iter=args.eta_em_steps,
                    )
                else:
                    eta_estimated = args.eta_fixed
                eta = eta_estimated
                gamma = 1.0 - eta
                loss = (
                    eta * F.softplus(-margin) - gamma * F.softplus(margin)
                ).mean() / (2.0 * eta - 1.0)
            else:
                effective_margin = margin
                if args.method == "margin_control" and args.control_mechanism == "relative_margin_clip":
                    if args.control_value >= 0:
                        median = current_margin_median(margin)
                        effective_margin = median + torch.clamp(margin - median, max=args.control_value)
                    update_margin_history(margin)
                dpo_loss = F.softplus(-effective_margin).mean()
                if args.method == "penalized_margin":
                    if args.penalty_variant == "global":
                        margin_penalty = margin.pow(2).mean()
                    elif args.penalty_variant == "selective":
                        margin_penalty = F.relu(margin.abs() - args.penalty_tau).pow(2).mean()
                    elif args.penalty_variant == "none":
                        margin_penalty = margin.new_tensor(0.0)
                    else:
                        raise ValueError(f"unknown penalty variant: {args.penalty_variant}")
                    loss = dpo_loss + args.penalty_lambda * margin_penalty
                else:
                    loss = dpo_loss
            last_loss = float(loss.detach().cpu())
            if args.method == "margin_control" and args.control_mechanism in {"per_example_grad_clip", "dp_sgd"}:
                if args.bsz != 1:
                    raise ValueError("per-example gradient clipping currently requires --bsz 1")
                accumulate_clipped_microbatch(loss)
            else:
                (loss / args.ga).backward()
            if (batch_index + 1) % args.ga == 0 or batch_index + 1 == len(loader):
                if args.method == "margin_control" and args.control_mechanism in {"per_example_grad_clip", "dp_sgd"}:
                    flush_accumulated_grads()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                for percent in sorted(
                    p for p, boundary in pending_checkpoints.items() if step >= boundary
                ):
                    pending_checkpoints.pop(percent)
                    write_checkpoint(percent, step)
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    # Any boundary the planned step count overshot still gets written, so the 100% checkpoint
    # exists even when the loop ends a step early.
    for percent in sorted(pending_checkpoints):
        pending_checkpoints.pop(percent)
        write_checkpoint(percent, step)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    selected_members_path = None
    if selected_member_rows is not None:
        selected_members_path = output / "selected_members.jsonl"
        if selected_members_path.exists():
            existing = read_jsonl(selected_members_path)
            if existing != selected_member_rows:
                raise FileExistsError(
                    "refusing to overwrite a different random-subset member set: "
                    f"{selected_members_path}"
                )
        else:
            write_jsonl(selected_members_path, selected_member_rows)
    model.save_pretrained(output, adapter_name="stage2")
    manifest_path = Path(args.manifest_out or output / "M2_manifest.json")
    dp_accounting = None
    if args.method == "margin_control" and args.control_mechanism == "dp_sgd":
        sample_rate = min(1.0, (args.bsz * args.ga) / max(1, len(rows)))
        dp_accounting = dp_sgd_epsilon_rdp(
            sample_rate=sample_rate,
            noise_multiplier=args.noise_multiplier,
            steps=step,
            delta=args.dp_delta,
        )
    manifest = {
        "base_model": args.model,
        "adapters": [{"name": "stage2", "path": str(output.resolve())}],
        "method": args.method,
        "params": {
            "fresh_from_base": True,
            "reference": "cached_base_logprobs",
            "ref_logps": str(Path(args.ref_logps).resolve()),
            "precision": {
                "policy_weights": (
                    POLICY_DTYPE if args.fp32 else ("bfloat16" if args.bf16 else "float16")
                ),
                "policy_logprobs": LOGPROB_DTYPE,
                "reference_cache": REFERENCE_DTYPE,
                "margin": LOGPROB_DTYPE,
            },
            "truncation": TRUNCATION_MODE,
            "max_len": args.max_len,
            # Recorded from the actual arguments: both are swept, and a hardcoded rank of 16
            # made every cell's provenance identical regardless of --lora_r.
            "lora_rank": args.lora_r,
            "epochs": args.epochs,
            "beta": args.beta,
            "lr": args.lr,
            "seed": args.seed,
            "n_train": len(rows),
            "data_sha256": file_sha256(args.data),
            "split_version": split_version_of(args.data),
            "source_bundle": os.environ.get("PDPO_SOURCE_BUNDLE"),
            "source_bundle_sha256": os.environ.get("PDPO_SOURCE_BUNDLE_SHA256"),
            "max_steps": args.max_steps,
            "planned_total_steps": planned_total_steps,
            "checkpoints": checkpoints_written,
            "eta_true": true_eta(args.epsilon),
            "eta_estimated": eta_estimated if args.method == "redpo" else None,
            "penalty_variant": args.penalty_variant if args.method == "penalized_margin" else None,
            "penalty_lambda": args.penalty_lambda if args.method == "penalized_margin" else None,
            "penalty_tau": args.penalty_tau if args.method == "penalized_margin" else None,
            "control_mechanism": args.control_mechanism if args.method == "margin_control" else None,
            "control_value": args.control_value if args.method == "margin_control" else None,
            "median_window": args.median_window if args.method == "margin_control" else None,
            "noise_multiplier": args.noise_multiplier if args.method == "margin_control" else None,
            "dp_delta": args.dp_delta if args.method == "margin_control" else None,
            "dp_accounting": dp_accounting,
            "selected_members_jsonl": (
                str(selected_members_path.resolve()) if selected_members_path else None
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "trainer_result.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "n": len(rows),
                "training_steps": step,
                "fresh_from_base": True,
                "reference": "cached_base_logprobs",
                "lora_rank": 16,
                "loss": last_loss,
                "eta_true": true_eta(args.epsilon),
                "eta_estimated": eta_estimated if args.method == "redpo" else None,
                "penalty_variant": args.penalty_variant if args.method == "penalized_margin" else None,
                "penalty_lambda": args.penalty_lambda if args.method == "penalized_margin" else None,
                "penalty_tau": args.penalty_tau if args.method == "penalized_margin" else None,
                "control_mechanism": args.control_mechanism if args.method == "margin_control" else None,
                "control_value": args.control_value if args.method == "margin_control" else None,
                "median_window": args.median_window if args.method == "margin_control" else None,
                "noise_multiplier": args.noise_multiplier if args.method == "margin_control" else None,
                "dp_delta": args.dp_delta if args.method == "margin_control" else None,
                "dp_accounting": dp_accounting,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main(default_method: str | None = None) -> int:
    args = build_parser(default_method).parse_args()
    if not args.method:
        raise ValueError("--method is required")
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("training data is empty")
    for row in rows:
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise ValueError(f"D2 row missing {key!r}")
    if args.tiny_test:
        return run_tiny(args, rows)
    return train_real(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
