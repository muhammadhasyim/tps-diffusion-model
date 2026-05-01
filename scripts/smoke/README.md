# OneOPES REPEX smoke scripts

## `run_oneopes_repex_scaleup.sh`

Staged **legacy-boltz** Hamiltonian replica exchange (2, 3, 4 replicas; optional 8
with `RUN_REPEX_STAGE8=1`) using `scripts/run_openmm_oneopes_repex.py`.

### Environment

- `PYTHONPATH` must include `src/python` (the script sets this from the repo root).
- `PLUMED_KERNEL` must point at a PLUMED **2.9.x** `libplumedKernel.so` built with
  the **opes** module (see `scripts/build_plumed_opes.sh`).
- `MOL_DIR` defaults to `~/.boltz/mols` (ligand CCD pickles such as `FZC.pkl`).

### Tunables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOPO_NPZ` / `FRAME_NPZ` | Case1 campaign artifacts under `artifacts/evaluation/` | Boltz inputs |
| `MOL_DIR` | `$HOME/.boltz/mols` | Boltz molecule cache |
| `OUT_BASE` | `artifacts/evaluation/repex_scaleup_smoke` | Parent directory for timestamped runs |
| `SAVE_OPES_EVERY` | `50000` | Must be **≥** every OPES `PACE` in the deck; auxiliary hydration OPES uses `--oneopes-water-pace` (default **40000**). |
| `DEPOSIT_PACE` | `500` | Main OPES deposit stride |
| `REPLICA_STEP_WORKERS` | `1` | Serial replica stepping avoids fragile multi-context CUDA stepping on **one** GPU. |
| `RUN_REPEX_STAGE8` | unset | Set to `1` to include the 8-replica stage (high VRAM). |

### Validation

After each stage, the script runs:

```bash
python scripts/smoke/validate_oneopes_repex_smoke.py <stage_out> --gpu-monitor-log <gpu_monitor_*.log>
```

## `validate_oneopes_repex_smoke.py`

Checks `exchange_log.csv`, `barrier_timing.jsonl`, per-replica `opes_states/`, and
optionally a `gpu_monitor_*.log` from `scripts/profile/run_gpu_monitor.sh`.

Core logic lives in `genai_tps.simulation.oneopes_repex_smoke_validate` (unit-tested).
