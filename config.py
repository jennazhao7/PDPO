"""Cost-control configuration for the PDPO single-GPU GCP pipeline.

Fill in the constants below, or export the matching environment variables.
The existing scripts/gcp/*.sh files source scripts/gcp/gcp.env; these Python
tools read environment variables so secrets/project IDs do not need to be
duplicated in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


PROJECT_ID = os.getenv("GCP_PROJECT", "<my-gcp-project>")
ZONE = os.getenv("GCP_ZONE", "<e.g. asia-northeast1-a>")
BUCKET = os.getenv("GCS_BUCKET", "gs://<my-bucket>")
CREDIT_CEILING_USD = float(os.getenv("CREDIT_CEILING_USD", "200"))
PREFERRED_CARD = os.getenv("PREFERRED_CARD", "L4")

DEFAULT_VM_NAME = os.getenv("VM_NAME", "pdpo-spot-runner")
DEFAULT_REGION = os.getenv("GCP_REGION", "-".join(ZONE.split("-")[:2]) if not ZONE.startswith("<") else "asia-northeast1")
BOOT_DISK_SIZE = os.getenv("BOOT_DISK_SIZE", "50GB")
IMAGE_FAMILY = os.getenv("IMAGE_FAMILY", "common-cu129-ubuntu-2204-nvidia-580")
IMAGE_PROJECT = os.getenv("IMAGE_PROJECT", "deeplearning-platform-release")


@dataclass(frozen=True)
class CardConfig:
    machine_type: str
    gpu_type: str
    gpu_count: int
    rough_spot_usd_per_hour: float
    rough_on_demand_usd_per_hour: float
    notes: str


CARD_CONFIGS: Dict[str, CardConfig] = {
    # Rough 2026 planning rates only; verify exact regional price before spend.
    # GCP flags verified against current docs:
    #   gcloud compute instances create
    #   --provisioning-model=SPOT
    #   --instance-termination-action=DELETE
    #   --accelerator=type=...,count=...
    # GCP GPU VM docs also note that Deep Learning VM images are not supported
    # on G2 VMs; launch.sh keeps the image configurable for that reason.
    "L4": CardConfig(
        machine_type="g2-standard-8",
        gpu_type="nvidia-l4",
        gpu_count=1,
        rough_spot_usd_per_hour=0.35,
        rough_on_demand_usd_per_hour=1.05,
        notes="Default: 24GB VRAM, good fit for Qwen2.5-3B LoRA once ref logps are cached.",
    ),
    "T4": CardConfig(
        machine_type="n1-standard-8",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
        rough_spot_usd_per_hour=0.14,
        rough_on_demand_usd_per_hour=0.52,
        notes="Cheapest viable 16GB option; requires cached reference logps / memory discipline.",
    ),
    "A100": CardConfig(
        machine_type="a2-highgpu-1g",
        gpu_type="nvidia-tesla-a100",
        gpu_count=1,
        rough_spot_usd_per_hour=1.10,
        rough_on_demand_usd_per_hour=3.75,
        notes="Do not use by default. On-demand A100 requires --force-a100 plus typed confirmation.",
    ),
}


def get_card_config(card: str | None = None) -> CardConfig:
    selected = (card or PREFERRED_CARD).upper()
    if selected not in CARD_CONFIGS:
        raise ValueError(f"Unknown GPU card {selected!r}; expected one of {sorted(CARD_CONFIGS)}")
    return CARD_CONFIGS[selected]


def validate_runtime_config(require_bucket: bool = True) -> None:
    missing = []
    if PROJECT_ID.startswith("<") or PROJECT_ID in {"", "your-project-id"}:
        missing.append("PROJECT_ID/GCP_PROJECT")
    if ZONE.startswith("<") or ZONE == "":
        missing.append("ZONE/GCP_ZONE")
    if require_bucket and (BUCKET.startswith("gs://<") or BUCKET in {"", "gs://your-bucket"}):
        missing.append("BUCKET/GCS_BUCKET")
    if CREDIT_CEILING_USD <= 0:
        missing.append("CREDIT_CEILING_USD")
    if missing:
        raise SystemExit("Fill config.py or export env vars before launch: " + ", ".join(missing))
