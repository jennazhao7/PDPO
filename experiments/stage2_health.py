#!/usr/bin/env python3
"""HEALTH.json writer and stop-condition enforcer for the Stage 2 sweep.

Called by the supervisor every poll. Writes ONE file with a one-word verdict as the first key,
then returns an exit code telling the supervisor whether to halt.

  exit 0   -> continue
  exit 10  -> STALLED   progress frozen while a GPU VM is RUNNING
  exit 11  -> STOPPED   spend ceiling reached
  exit 12  -> FAILED    launch churn: repeated create failures that are NOT external blocking

WHY THE STALL CHECK EXISTS
  The existing circuit breaker only fires on run_status=="failed", which a wedged process never
  writes -- it just stops making progress with the VM still RUNNING and the GPU still billing.
  That is the exact gap that let the token-concentration job cycle for hours. This closes it.

WHY CHURN EXEMPTS EXTERNAL BLOCKING
  GPUS_ALL_REGIONS is capped at 1 globally on this project, so while any OTHER GPU VM exists
  (for example the E9 generation VM) every create legitimately fails. Counting those as churn
  would halt the sweep for the wrong reason, so a failure is only churn when no other GPU VM is
  holding the quota. Waiting on a busy quota is reported as HEALTHY with a `blocked_by` note.

Spend is derived from GCP's own operation records for the sweep VM -- insert/delete pairs summed
into VM-hours -- not from the ledger's row timestamps, which include long gaps when nothing ran.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Verified against GCP's own SKU catalog for asia-east1 (g2-standard-8 Spot):
# L4 GPU 0.37180 + 8 vCPU x 0.01659 + 32 GiB x 0.001943 = 0.5669 USD/hr
SPOT_USD_PER_HOUR = 0.5669
# Raised from 3.0. Training produces NO ledger rows: eps=0 e2 is ~3.8 h of training before its
# first commit and e3 is ~5.7 h, so a 3 h time-based arm would have deleted the VM mid-training
# and destroyed unrecoverable work. 8 h clears e3's first commit (~6.9 h) with margin, and is
# only a backstop -- the log-silence arm below is what actually detects a wedge.
STALL_HOURS = 8.0
# A healthy run writes startup.log continuously (observed going 1479.1 -> 0.6 min). A wedged
# process stops. STALL requires BOTH arms, so it cannot fire during legitimate training.
LOG_SILENT_MINUTES = 30.0
# Raised 90.0 -> 110.0 on 2026-08-23. At the measured 2.8 h/cell, the 48 cells remaining after
# the eps=0 arm project to ~$76 more (~$98 total), which would trip the ceiling around cell 55 of
# 60 and truncate the eps=2.0 arm. $20 of headroom is cheaper than another stop-fix-verify cycle,
# and eval batching is explicitly NOT being attempted -- the sweep is not to be stopped again for
# optimization.
SPEND_CEILING_USD = 110.0
CHURN_LIMIT = 5


def sh(args: List[str], timeout: int = 120) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except Exception:
        return ""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_ts(value: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def gpu_vm_hours(project: str, vm_name: str) -> float:
    """Sum uptime from insert/delete operation pairs. GCP's own records, not our bookkeeping."""
    # BOUNDED, and value-formatted rather than JSON. The unbounded --format=json form pulled
    # thousands of operations into memory on a 2 GB e2-small and killed the supervisor outright.
    # 400 most-recent operations covers far more than this VM's lifetime.
    raw = sh(["gcloud", "compute", "operations", "list", f"--project={project}",
              "--sort-by=~insertTime", "--limit=400",
              "--filter=targetLink~instances/" + vm_name,
              "--format=value(insertTime,operationType,targetLink,error.errors[0].code)"],
             timeout=120)
    if not raw:
        return 0.0
    events = []
    for line in raw.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        if not parts[2].endswith("/" + vm_name):
            continue
        ts = parse_ts(parts[0])
        if ts is None:
            continue
        had_error = len(parts) >= 4 and parts[3] not in ("", "None")
        events.append((ts, parts[1], had_error))
    events.sort()
    hours, open_at = 0.0, None
    for ts, typ, err in events:
        if typ == "insert" and not err:
            if open_at is None:
                open_at = ts
        elif typ == "delete" and open_at is not None:
            hours += (ts - open_at).total_seconds() / 3600.0
            open_at = None
    if open_at is not None:  # still running
        hours += (utcnow() - open_at).total_seconds() / 3600.0
    return hours


def other_gpu_vms(project: str, own: str) -> List[str]:
    """GPU-bearing instances other than ours -- these hold the global GPU quota."""
    raw = sh(["gcloud", "compute", "instances", "list", f"--project={project}",
              "--format=value(name,guestAccelerators.acceleratorCount)"])
    names = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts or parts[0] == own:
            continue
        if len(parts) > 1 and parts[1] not in ("", "0"):
            names.append(parts[0])
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--vm-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--consecutive-launch-failures", type=int, default=0)
    ap.add_argument("--relaunches-used", type=int, default=0)
    ap.add_argument("--max-relaunches", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    prefix = f"{args.bucket.rstrip('/')}/experiments/stage2_rr"
    now = utcnow()

    # The ledger and status live in the BUCKET -- the GPU VM syncs them there on every commit.
    # The supervisor VM has no local copy (its bundle does not include them), so reading
    # out_dir alone would report progress=0 forever and stall detection would never fire.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "ledger.jsonl"):
        sh(["gcloud", "storage", "cp", f"{prefix}/{name}", str(out_dir / name)], timeout=90)

    status: Dict[str, Any] = {}
    sp = out_dir / "status.json"
    if sp.is_file():
        try:
            status = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            status = {}

    ledger: List[Dict[str, Any]] = []
    lp = out_dir / "ledger.jsonl"
    if lp.is_file():
        for line in lp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ledger.append(json.loads(line))
                except Exception:
                    pass

    progress = len(ledger)
    expected = status.get("expected_cells", 60)
    last_cell = ledger[-1].get("key") if ledger else None
    last_ts_raw = ledger[-1].get("timestamp") if ledger else None
    last_ts = parse_ts(last_ts_raw) if last_ts_raw else None
    mins_since_commit = ((now - last_ts).total_seconds() / 60.0) if last_ts else None

    # GPU VM state
    rows = sh(["gcloud", "compute", "instances", "list", f"--project={args.project}",
               f"--filter=name={args.vm_name}",
               "--format=value(name,zone.basename(),status,creationTimestamp)"])
    vm_name = vm_zone = vm_state = None
    vm_created = None
    if rows:
        parts = rows.split()
        if len(parts) >= 3:
            vm_name, vm_zone, vm_state = parts[0], parts[1], parts[2]
        if len(parts) >= 4:
            vm_created = parse_ts(parts[3])
    vm_age_min = ((now - vm_created).total_seconds() / 60.0) if vm_created else None

    startup_age = None
    ls = sh(["gcloud", "storage", "ls", "-l", f"{prefix}/startup.log"])
    for line in ls.splitlines():
        bits = line.split()
        if len(bits) >= 2 and bits[1].endswith("Z"):
            t = parse_ts(bits[1])
            if t:
                startup_age = (now - t).total_seconds() / 60.0
            break

    hours = gpu_vm_hours(args.project, args.vm_name)
    spend = hours * SPOT_USD_PER_HOUR
    blockers = other_gpu_vms(args.project, args.vm_name)

    # Stall is measured from the LATER of (last commit, this VM's creation). Using absolute time
    # since the last ledger row would fire the instant a VM launches after any gap longer than
    # the threshold -- and there is a 27-hour gap right now from the stop-and-optimize pause, so
    # a fresh launch would have been deleted on its first poll.
    stall_ref = None
    if vm_created is not None and last_ts is not None:
        stall_ref = max(vm_created, last_ts)
    elif vm_created is not None:
        stall_ref = vm_created
    elif last_ts is not None:
        stall_ref = last_ts
    stall_min = ((now - stall_ref).total_seconds() / 60.0) if stall_ref else None
    # An unreadable log age must NOT satisfy the silence arm: a failed read should never be the
    # thing that deletes a healthy training run.
    log_silent_min = startup_age if startup_age is not None else 0.0

    # ---- verdict and halt decision -----------------------------------------------------------
    verdict = "HEALTHY"
    halt = 0
    reason = None
    # A frozen writer previously left `verdict: HEALTHY` in place for 24 hours while nothing ran.
    # The reader cannot distinguish that from real health, so the file states its own freshness
    # and the morning check compares timestamp_utc against this budget.
    stale_after_min = 12.0

    if spend >= SPEND_CEILING_USD:
        verdict, halt = "STOPPED", 11
        reason = (f"spend ceiling: ${spend:.2f} >= ${SPEND_CEILING_USD:.2f} "
                  f"({hours:.2f} GPU-hr x ${SPOT_USD_PER_HOUR}/hr)")
    elif (vm_state == "RUNNING"
          and stall_min is not None and stall_min >= STALL_HOURS * 60
          and log_silent_min >= LOG_SILENT_MINUTES):
        verdict, halt = "STALLED", 10
        reason = (f"no commit in {stall_min:.0f} min of this VM's life (>= {STALL_HOURS}h) AND "
                  f"startup.log silent {log_silent_min:.0f} min (>= {LOG_SILENT_MINUTES}) "
                  f"while {args.vm_name} is RUNNING -- wedged, not training")
    elif args.consecutive_launch_failures >= CHURN_LIMIT and not blockers:
        verdict, halt = "FAILED", 12
        reason = (f"{args.consecutive_launch_failures} consecutive all-zone create failures "
                  "with no other GPU VM holding the global quota")
    elif progress >= expected:
        verdict = "STOPPED"
        reason = "sweep complete"

    health = {
        "verdict": verdict,
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "progress": f"{progress}/{expected}",
        "last_cell_committed": last_cell,
        "last_cell_committed_at": last_ts_raw,
        "minutes_since_last_commit": (round(mins_since_commit, 1)
                                      if mins_since_commit is not None else None),
        "gpu_vm_name": vm_name or args.vm_name,
        "gpu_vm_zone": vm_zone,
        "gpu_vm_state": vm_state or "ABSENT",
        "gpu_vm_age_minutes": (round(vm_age_min, 1) if vm_age_min is not None else None),
        "minutes_without_commit_this_vm": (round(stall_min, 1)
                                          if stall_min is not None else None),
        "minutes_since_startup_log_changed": (round(startup_age, 1)
                                              if startup_age is not None else None),
        "run_status": status.get("status"),
        "relaunches_used": args.relaunches_used,
        "max_relaunches": args.max_relaunches,
        "consecutive_launch_failures": args.consecutive_launch_failures,
        "cumulative_gpu_hours": round(hours, 3),
        "estimated_spend_usd": round(spend, 2),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "stall_threshold_hours": STALL_HOURS,
        "stall_log_silent_threshold_minutes": LOG_SILENT_MINUTES,
        "stall_requires": ("minutes_without_commit_this_vm >= stall_threshold_hours*60 AND "
                           "minutes_since_startup_log_changed >= "
                           "stall_log_silent_threshold_minutes"),
        "churn_limit": CHURN_LIMIT,
        # Waiting on a quota another VM holds is HEALTHY, not churn. Named so it is obvious.
        "blocked_by_other_gpu_vms": blockers or None,
        "stale_after_minutes": stale_after_min,
        "reader_note": ("If timestamp_utc is older than stale_after_minutes, treat this file as "
                        "STALE regardless of verdict -- the writer has stopped."),
        "halt_reason": reason,
        "halting": bool(halt),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "HEALTH.json").write_text(
        json.dumps(health, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    sh(["gcloud", "storage", "cp", str(out_dir / "HEALTH.json"), f"{prefix}/HEALTH.json"])
    print(f"[health] {verdict} progress={progress}/{expected} "
          f"spend=${spend:.2f} vm={vm_state or 'ABSENT'} "
          f"blocked_by={blockers or '-'} halt={halt}")
    return halt


if __name__ == "__main__":
    raise SystemExit(main())
