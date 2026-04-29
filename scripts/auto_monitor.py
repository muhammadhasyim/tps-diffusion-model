#!/usr/bin/env python3
"""Periodic monitoring companion for auto_pipeline.py.

Runs LOCALLY. Every --interval seconds:
  1. Creates a lightweight CPU sandbox (no GPU)
  2. Runs monitor_progress.py to generate the dashboard
  3. Downloads the dashboard PNG and stats JSON locally
  4. Terminates the sandbox
  5. Opens the dashboard in the Axon viewer (optional)

Exits when the pipeline reaches the target or is interrupted.

Usage:
    python scripts/auto_monitor.py --interval 600   # every 10 min
    python scripts/auto_monitor.py --once            # single snapshot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODAL_TOOLS = "modal-tools"
BASE_DIR = "/mnt/shared/26-04-22-tps-diffusion/opes_wdsm_12k_long"
TPS_REPO = "/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model"
IMAGE = "chemistry-py-3.12"  # CPU-only image — no GPU needed for plotting
LOCAL_OUTPUT = Path(__file__).resolve().parents[1] / "monitoring_output"


def run_cli(args: list[str], timeout: int = 60) -> dict:
    result = subprocess.run(
        [MODAL_TOOLS] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal-tools failed: {result.stderr[:500]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}


def sandbox_exec(sandbox_id: str, code: str, timeout: int = 300) -> dict:
    result = subprocess.run(
        [MODAL_TOOLS, "sandbox", "execute-python", sandbox_id,
         "--code", code, "--timeout", str(timeout), "--json"],
        capture_output=True, text=True, timeout=timeout + 60,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout, "stderr": result.stderr}


def create_cpu_sandbox() -> str:
    result = run_cli([
        "sandbox", "create",
        "--image", IMAGE,
        "--timeout", "600",
        "--json",
    ], timeout=120)
    sid = result.get("sandbox_id")
    if not sid:
        raise RuntimeError(f"Failed to create sandbox: {result}")
    return sid


def terminate_sandbox(sandbox_id: str) -> None:
    try:
        run_cli(["sandbox", "terminate", sandbox_id], timeout=30)
    except Exception:
        pass


def download_file(remote: str, local: str) -> None:
    subprocess.run(
        [MODAL_TOOLS, "volume", "download", "--remote-path", remote, "--local-path", local],
        capture_output=True, text=True, timeout=60,
    )


def run_monitor_snapshot(target_steps: int) -> dict | None:
    sandbox_id = None
    try:
        print(f"[monitor] Creating CPU sandbox...")
        sandbox_id = create_cpu_sandbox()

        print(f"[monitor] Installing genai-tps (minimal)...")
        sandbox_exec(sandbox_id, f"""
import subprocess
subprocess.run(["pip", "install", "-e", "{TPS_REPO}/[boltz]"],
               capture_output=True, text=True, timeout=300)
print("installed")
""", timeout=360)

        print(f"[monitor] Generating dashboard...")
        r = sandbox_exec(sandbox_id, f"""
import sys
sys.path.insert(0, "{TPS_REPO}/scripts")
from monitor_progress import generate_dashboard
from pathlib import Path
import json

stats = generate_dashboard(
    Path("{BASE_DIR}"),
    Path("{BASE_DIR}/monitoring"),
    target_steps={target_steps},
)
print(json.dumps(stats))
""", timeout=300)

        stdout = r.get("stdout", "")
        print(f"[monitor] {stdout[-500:]}")

        terminate_sandbox(sandbox_id)

        try:
            stats_line = [l for l in stdout.strip().split('\n') if l.startswith('{')]
            if stats_line:
                return json.loads(stats_line[-1])
        except (json.JSONDecodeError, IndexError):
            pass
        return None

    except Exception as e:
        print(f"[monitor] ERROR: {e}")
        if sandbox_id:
            terminate_sandbox(sandbox_id)
        return None


def download_outputs() -> None:
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    for fname in ["opes_dashboard.png", "monitor_stats.json"]:
        remote = f"{BASE_DIR}/monitoring/{fname}"
        local = str(LOCAL_OUTPUT / fname)
        try:
            download_file(remote, local)
            print(f"[monitor] Downloaded {fname}")
        except Exception as e:
            print(f"[monitor] Download failed for {fname}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Periodic OPES-TPS monitoring")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between snapshots")
    parser.add_argument("--target-steps", type=int, default=12000)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  OPES-TPS Monitor")
    print(f"  Interval: {'once' if args.once else f'{args.interval}s'}")
    print(f"  Target: {args.target_steps} steps")
    print(f"  Local output: {LOCAL_OUTPUT}")
    print(f"{'='*50}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Snapshot #{iteration} ({time.strftime('%H:%M:%S')}) ---")

        stats = run_monitor_snapshot(args.target_steps)
        download_outputs()

        if stats:
            n = stats.get("n_samples", 0)
            target = stats.get("target_steps", args.target_steps)
            neff_frac = stats.get("n_eff_fraction", 0)
            print(f"[monitor] Progress: {n}/{target} ({n/target*100:.1f}%) | "
                  f"N_eff: {neff_frac*100:.1f}%")

            if n >= target:
                print(f"\n[monitor] TARGET REACHED ({n} steps). Pipeline complete.")
                break

        if args.once:
            break

        print(f"[monitor] Next snapshot in {args.interval}s...")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[monitor] Interrupted.")
            break


if __name__ == "__main__":
    main()
