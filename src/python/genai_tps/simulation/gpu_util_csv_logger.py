"""Background CSV logger for NVIDIA GPU utilization via ``nvidia-smi``.

For higher-rate logs or DCGM-based sampling from a separate process, see
``scripts/profile/run_gpu_monitor.sh``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

__all__ = ["GpuUtilCsvLogger", "nvidia_smi_query_line"]


def nvidia_smi_query_line(gpu_index: int) -> str | None:
    """Return one CSV data line from ``nvidia-smi`` or ``None`` on failure.

    Columns (``nounits``): utilization.gpu %, memory.used MiB, memory.total MiB,
    power.draw W (may be empty on some drivers).
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    cmd = [
        exe,
        f"-i={int(gpu_index)}",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    line = proc.stdout.strip().splitlines()[0].strip()
    return line or None


class GpuUtilCsvLogger:
    """Sample ``nvidia-smi`` every *interval_s* and append rows to a CSV file.

    Intended to run in a daemon thread while MD holds the GPU; call
    :meth:`stop` before reading the file for a complete flush.
    """

    def __init__(
        self,
        out_csv: Path,
        interval_s: float,
        gpu_index: int,
        *,
        query_fn: Callable[[int], str | None] | None = None,
    ) -> None:
        self._csv = Path(out_csv).expanduser().resolve()
        self._interval = max(0.02, float(interval_s))
        self._gpu = int(gpu_index)
        self._query = query_fn or nvidia_smi_query_line
        self._uses_nvidia_smi = query_fn is None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def csv_path(self) -> Path:
        return self._csv

    def start(self) -> bool:
        """Start the background thread.

        Returns ``False`` if the default ``nvidia-smi`` query is used but the
        executable is not on ``PATH``. Custom ``query_fn`` (e.g. in unit tests)
        does not require ``nvidia-smi``.
        """
        if self._uses_nvidia_smi and shutil.which("nvidia-smi") is None:
            return False
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-util-csv-logger",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Signal stop and wait briefly for the writer thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval * 5))

    def _run(self) -> None:
        self._csv.parent.mkdir(parents=True, exist_ok=True)
        with self._csv.open("w", encoding="utf-8", newline="") as fh:
            fh.write(
                "unix_time_s,gpu_index,util_gpu_pct,memory_used_mib,"
                "memory_total_mib,power_draw_w\n"
            )
            fh.flush()
            while not self._stop.is_set():
                t_loop = time.perf_counter()
                ts = time.time()
                raw = self._query(self._gpu)
                if raw:
                    parts = [p.strip() for p in raw.split(",")]
                    while len(parts) < 4:
                        parts.append("")
                    util, mused, mtot, pwr = parts[0], parts[1], parts[2], parts[3]
                    fh.write(
                        f"{ts:.6f},{self._gpu},{util},{mused},{mtot},{pwr}\n"
                    )
                else:
                    fh.write(f"{ts:.6f},{self._gpu},,,,\n")
                fh.flush()
                elapsed = time.perf_counter() - t_loop
                wait = max(0.0, self._interval - elapsed)
                if self._stop.wait(wait):
                    break
