#!/usr/bin/env python3
"""Shared matched-design utilities with no mandatory ML imports."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MATCHED_DEFAULTS = {
    "beta": 0.5,
    "lr": 2.5e-5,
    "epochs": 3,
    "max_steps": -1,
    "bsz": 1,
    "ga": 16,
    "max_len": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
}

REQUIRED_TRAINER_RESULT_FIELDS = {
    "method": str,
    "n": int,
    "training_steps": int,
    "fresh_from_base": bool,
    "reference": str,
    "lora_rank": int,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_key(row: dict[str, Any]) -> str:
    if row.get("pair_id") is not None:
        return f"id:{row['pair_id']}"
    payload = "\0".join((str(row["prompt"]), str(row["chosen"]), str(row["rejected"])))
    return f"sha256:{text_hash(payload)}"


def true_eta(eps: float) -> float:
    return math.exp(eps) / (1.0 + math.exp(eps))


def softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def robust_channel_losses(logits: Sequence[float], eta: float) -> list[float]:
    """Unbiased robust-DPO loss for RR reliability eta=P(label is retained)."""
    if not 0.5 < eta < 1.0:
        raise ValueError(f"eta must be in (0.5, 1), got {eta}")
    denominator = 2.0 * eta - 1.0
    gamma = 1.0 - eta
    return [
        (eta * softplus(-float(logit)) - gamma * softplus(float(logit))) / denominator
        for logit in logits
    ]


def rdpo_loss(logits: Sequence[float], eps: float) -> float:
    losses = robust_channel_losses(logits, true_eta(eps))
    return sum(losses) / max(1, len(losses))


def redpo_loss(logits: Sequence[float], eta: float) -> float:
    """RE-DPO channel-corrected loss after eta is estimated (or fixed for tests)."""
    losses = robust_channel_losses(logits, eta)
    return sum(losses) / max(1, len(losses))


def penalized_margin_term(
    margins: Sequence[float],
    variant: str,
    tau: float = 0.0,
) -> float:
    """Mean margin-control penalty on post-beta DPO margins.

    The ``global`` variant penalizes all margin magnitude.  The ``selective``
    variant leaves margins inside [-tau, tau] untouched and penalizes only the
    excess magnitude.
    """
    if variant == "none":
        return 0.0
    if variant not in {"global", "selective"}:
        raise ValueError(f"unknown penalty variant: {variant}")
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    if not margins:
        return 0.0
    penalties: list[float] = []
    for value in margins:
        magnitude = abs(float(value))
        if variant == "global":
            penalties.append(magnitude * magnitude)
        else:
            excess = max(0.0, magnitude - tau)
            penalties.append(excess * excess)
    return sum(penalties) / len(penalties)


def relative_margin_clip_values(
    margins: Sequence[float],
    median: float,
    cap: float,
) -> list[float]:
    """Cap positive excess over a detached population median.

    A negative cap is treated as a no-op.  This mirrors the trainer's
    relative-margin clipping mechanism in a lightweight, testable form.
    """
    if cap < 0:
        return [float(value) for value in margins]
    clipped: list[float] = []
    for value in margins:
        excess = float(value) - float(median)
        clipped.append(float(median) + min(excess, float(cap)))
    return clipped


def dp_sgd_epsilon_rdp(
    *,
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    delta: float,
    orders: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Conservative RDP accountant for Poisson-sampled Gaussian DP-SGD."""
    if orders is None:
        orders = tuple(range(2, 65)) + (128, 256)
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if noise_multiplier <= 0:
        return {
            "epsilon_g": None,
            "epsilon_g_infinite": True,
            "delta": delta,
            "sample_rate": sample_rate,
            "noise_multiplier": noise_multiplier,
            "steps": steps,
            "order": None,
            "accountant": "rdp_subsampled_gaussian_integer_orders",
        }

    best_epsilon = float("inf")
    best_order: int | None = None
    log_q = math.log(sample_rate)
    log_1mq = math.log1p(-sample_rate) if sample_rate < 1.0 else float("-inf")
    for alpha in orders:
        if alpha <= 1:
            continue
        terms: list[float] = []
        for i in range(alpha + 1):
            log_binom = math.log(math.comb(alpha, i))
            q_term = i * log_q
            one_minus_term = (alpha - i) * log_1mq
            gaussian_term = (i * i - i) / (2.0 * noise_multiplier * noise_multiplier)
            terms.append(log_binom + q_term + one_minus_term + gaussian_term)
        max_term = max(terms)
        log_a = max_term + math.log(sum(math.exp(term - max_term) for term in terms))
        rdp = steps * log_a / (alpha - 1.0)
        epsilon = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_order = alpha
    return {
        "epsilon_g": best_epsilon,
        "epsilon_g_infinite": False,
        "delta": delta,
        "sample_rate": sample_rate,
        "noise_multiplier": noise_multiplier,
        "steps": steps,
        "order": best_order,
        "accountant": "rdp_subsampled_gaussian_integer_orders",
    }


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def estimate_eta_em(
    logits: Sequence[float],
    initial_eta: float = 0.75,
    max_iter: int = 50,
    tol: float = 1e-8,
    lower: float = 0.5001,
    upper: float = 0.9999,
) -> tuple[float, list[float], int]:
    """Estimate RR reliability by EM using policy clean-label probabilities."""
    if not logits:
        raise ValueError("cannot estimate eta from an empty batch")
    eta = min(upper, max(lower, float(initial_eta)))
    responsibilities: list[float] = []
    for iteration in range(1, max_iter + 1):
        responsibilities = []
        for logit in logits:
            clean_prob = sigmoid(float(logit))
            numerator = eta * clean_prob
            denominator = numerator + (1.0 - eta) * (1.0 - clean_prob)
            responsibilities.append(numerator / max(denominator, 1e-12))
        updated = min(upper, max(lower, sum(responsibilities) / len(responsibilities)))
        if abs(updated - eta) <= tol:
            return updated, responsibilities, iteration
        eta = updated
    return eta, responsibilities, max_iter


def deterministic_subset_indices(n: int, fraction: float, seed: int) -> list[int]:
    if n <= 0:
        return []
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return sorted(indices[: max(1, int(round(n * fraction)))])


_UNSET = object()


def load_reference_cache(
    cache_path: str | Path,
    rows: Sequence[dict[str, Any]],
    expected_base_model: str | None = None,
    expected_scoring_dtype: str | None = None,
    expected_truncation: str | None = None,
    expected_max_len: int | None = None,
    expected_rr_eps: Any = _UNSET,
    expected_cache_role: str | None = None,
) -> tuple[list[float], list[float]]:
    """Reference logprob sums for `rows`, validated row by row. Fail-closed.

    `expected_rr_eps` and `expected_cache_role` state guarantees that were previously only
    incidental. Cross-arm reuse was already refused, but only because SOME row's orientation
    differs, and the check is per-row: a subset with zero differing rows would have loaded. These
    two assertions make arm identity and cache purpose explicit rather than emergent.

    A cache written before these fields existed carries neither, so a caller that passes an
    expectation will be told to rebuild rather than silently accepting an unidentified cache.
    """
    cached = read_jsonl(cache_path)
    by_key: dict[str, dict[str, Any]] = {}
    for record in cached:
        key = str(record.get("pair_key") or "")
        if not key:
            raise ValueError(f"cache row lacks pair_key: {cache_path}")
        if key in by_key:
            raise ValueError(f"duplicate pair_key in cache: {key}")
        by_key[key] = record

    chosen: list[float] = []
    rejected: list[float] = []
    for row in rows:
        key = pair_key(row)
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"reference cache missing D2 pair {key}")
        if expected_base_model and record.get("base_model") != expected_base_model:
            raise ValueError(
                f"cache base mismatch for {key}: {record.get('base_model')} != {expected_base_model}"
            )
        if (
            expected_scoring_dtype
            and record.get("scoring_dtype") != expected_scoring_dtype
        ):
            raise ValueError(
                f"cache dtype mismatch for {key}: "
                f"{record.get('scoring_dtype')} != {expected_scoring_dtype}"
            )
        if expected_truncation and record.get("truncation") != expected_truncation:
            raise ValueError(
                f"cache truncation mismatch for {key}: "
                f"{record.get('truncation')} != {expected_truncation}"
            )
        if expected_max_len is not None and record.get("max_len") != expected_max_len:
            raise ValueError(
                f"cache max_len mismatch for {key}: "
                f"{record.get('max_len')} != {expected_max_len}"
            )
        if expected_rr_eps is not _UNSET and record.get("rr_eps") != expected_rr_eps:
            raise ValueError(
                f"cache arm mismatch for {key}: rr_eps={record.get('rr_eps')!r} != "
                f"{expected_rr_eps!r}. A cache without rr_eps predates this check; rebuild it."
            )
        if expected_cache_role and record.get("cache_role") != expected_cache_role:
            raise ValueError(
                f"cache role mismatch for {key}: {record.get('cache_role')!r} != "
                f"{expected_cache_role!r}. Refusing to use a cache built for another purpose."
            )
        if record.get("prompt_hash") != text_hash(str(row["prompt"])):
            raise ValueError(f"prompt mismatch for cached pair {key}")
        if record.get("chosen_hash") != text_hash(str(row["chosen"])):
            raise ValueError(f"chosen orientation mismatch for cached pair {key}")
        if record.get("rejected_hash") != text_hash(str(row["rejected"])):
            raise ValueError(f"rejected orientation mismatch for cached pair {key}")
        chosen.append(float(record["chosen_ref_logp"]))
        rejected.append(float(record["rejected_ref_logp"]))
    return chosen, rejected


def load_reference_cache_detail(
    cache_path: str | Path,
    rows: Sequence[dict[str, Any]],
    expected_base_model: str | None = None,
    expected_scoring_dtype: str | None = None,
    expected_truncation: str | None = None,
    expected_max_len: int | None = None,
    expected_rr_eps: Any = _UNSET,
    expected_cache_role: str | None = None,
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """As load_reference_cache, but also returns the PER-TOKEN reference logprobs.

    eval_privacy_audit persists per-token logprobs for all four series (policy and reference,
    chosen and rejected) and the ledger records that artifact's row count and sha256. Serving the
    reference side from a sums-only cache would halve n_token_logprobs and change the artifact
    shape, breaking comparability with the four already-committed cells. So the cache carries the
    per-token arrays too, stored exactly as the evaluator would have produced them, and this
    loader hands them back verbatim.

    The stored sum is NOT recomputed from the per-token list: the evaluator's sum comes from a
    torch float32 reduction while a Python sum over the list is float64, so the two differ in the
    last bits. Both are stored, and their agreement is checked only loosely, as corruption
    detection.
    """
    cached = read_jsonl(cache_path)
    by_key: dict[str, dict[str, Any]] = {}
    for record in cached:
        key = str(record.get("pair_key") or "")
        if not key:
            raise ValueError(f"cache row lacks pair_key: {cache_path}")
        if key in by_key:
            raise ValueError(f"duplicate pair_key in cache: {key}")
        by_key[key] = record

    chosen: list[float] = []
    rejected: list[float] = []
    chosen_tokens: list[list[float]] = []
    rejected_tokens: list[list[float]] = []
    for row in rows:
        key = pair_key(row)
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"reference cache missing D2 pair {key}")
        if expected_base_model and record.get("base_model") != expected_base_model:
            raise ValueError(
                f"cache base mismatch for {key}: {record.get('base_model')} != {expected_base_model}"
            )
        if expected_scoring_dtype and record.get("scoring_dtype") != expected_scoring_dtype:
            raise ValueError(
                f"cache dtype mismatch for {key}: "
                f"{record.get('scoring_dtype')} != {expected_scoring_dtype}"
            )
        if expected_truncation and record.get("truncation") != expected_truncation:
            raise ValueError(
                f"cache truncation mismatch for {key}: "
                f"{record.get('truncation')} != {expected_truncation}"
            )
        if expected_max_len is not None and record.get("max_len") != expected_max_len:
            raise ValueError(
                f"cache max_len mismatch for {key}: "
                f"{record.get('max_len')} != {expected_max_len}"
            )
        if expected_rr_eps is not _UNSET and record.get("rr_eps") != expected_rr_eps:
            raise ValueError(
                f"cache arm mismatch for {key}: rr_eps={record.get('rr_eps')!r} != "
                f"{expected_rr_eps!r}. A cache without rr_eps predates this check; rebuild it."
            )
        if expected_cache_role and record.get("cache_role") != expected_cache_role:
            raise ValueError(
                f"cache role mismatch for {key}: {record.get('cache_role')!r} != "
                f"{expected_cache_role!r}. Refusing to use a cache built for another purpose."
            )
        if record.get("prompt_hash") != text_hash(str(row["prompt"])):
            raise ValueError(f"prompt mismatch for cached pair {key}")
        if record.get("chosen_hash") != text_hash(str(row["chosen"])):
            raise ValueError(f"chosen orientation mismatch for cached pair {key}")
        if record.get("rejected_hash") != text_hash(str(row["rejected"])):
            raise ValueError(f"rejected orientation mismatch for cached pair {key}")
        for field in ("chosen_ref_per_token", "rejected_ref_per_token"):
            if not isinstance(record.get(field), list):
                raise ValueError(
                    f"cache row {key} lacks {field}; this cache cannot serve the per-token "
                    "artifact. Rebuild with experiments/build_eval_ref_cache.py."
                )
        chosen_sum = float(record["chosen_ref_logp"])
        rejected_sum = float(record["rejected_ref_logp"])
        chosen_pt = [float(v) for v in record["chosen_ref_per_token"]]
        rejected_pt = [float(v) for v in record["rejected_ref_per_token"]]
        # Corruption check only -- see the docstring on why this is loose, not exact.
        for label, total, values in (
            ("chosen", chosen_sum, chosen_pt),
            ("rejected", rejected_sum, rejected_pt),
        ):
            if values and abs(sum(values) - total) > 1e-2:
                raise ValueError(
                    f"cache row {key} {label}: per-token sum {sum(values)} disagrees with stored "
                    f"sum {total} beyond corruption tolerance"
                )
        chosen.append(chosen_sum)
        rejected.append(rejected_sum)
        chosen_tokens.append(chosen_pt)
        rejected_tokens.append(rejected_pt)
    return chosen, rejected, chosen_tokens, rejected_tokens


def validate_trainer_result(result: dict[str, Any]) -> None:
    for key, expected in REQUIRED_TRAINER_RESULT_FIELDS.items():
        if key not in result or not isinstance(result[key], expected):
            raise ValueError(f"invalid trainer result field {key!r}")
    if result["n"] <= 0 or result["training_steps"] <= 0:
        raise ValueError("n and training_steps must be positive")
    if not result["fresh_from_base"]:
        raise ValueError("matched design requires fresh_from_base=true")
    if result["reference"] != "cached_base_logprobs":
        raise ValueError("matched design requires cached base-model reference logprobs")
    if result["lora_rank"] != 16:
        raise ValueError("matched design requires rank 16")


def write_tiny_artifacts(
    out: str | Path,
    method: str,
    rows: Sequence[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "method": method,
        "n": len(rows),
        "training_steps": 1,
        "fresh_from_base": True,
        "reference": "cached_base_logprobs",
        "lora_rank": 16,
        "timestamp": utc_now(),
    }
    if extra:
        result.update(extra)
    validate_trainer_result(result)
    (output / "trainer_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "base_model": "tiny-fixture",
        "adapters": [{"name": "stage2", "path": str(output.resolve())}],
        "method": method,
        "params": {
            "fresh_from_base": True,
            "reference": "cached_base_logprobs",
            "lora_rank": 16,
            "tiny_test": True,
        },
    }
    (output / "M2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
