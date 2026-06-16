#!/usr/bin/env python3
import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_utilization() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [int(line.strip()) for line in out.splitlines() if line.strip()]
    return max(values) if values else 0


def shutdown(log_path: Path, dry_run: bool) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] idle threshold exceeded; shutting down\n")
    if dry_run:
        return
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Shut down the VM after sustained low GPU utilization.")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--idle-minutes", type=int, default=10)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--log", default="idle_watchdog.log")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    low_since = None
    while True:
        try:
            util = gpu_utilization()
        except Exception as exc:
            util = 0
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{utc_now()}] nvidia-smi error treated as idle: {exc}\n")
        now = time.time()
        if util < args.threshold:
            if low_since is None:
                low_since = now
        else:
            low_since = None
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now()}] gpu_util={util} low_since={low_since}\n")
        if low_since is not None and now - low_since >= args.idle_minutes * 60:
            shutdown(log_path, args.dry_run)
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
