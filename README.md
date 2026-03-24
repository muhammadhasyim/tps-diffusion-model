# genai-tps

**genai-tps** is a Python library for **transition path sampling (TPS)** and related path-ensemble methods applied to **generative models** (e.g. diffusion samplers). It **vendors [OpenPathSampling](https://github.com/openpathsampling/openpathsampling)** (MIT) as the sampling driver—ensembles, moves, networks—so you do not need a separate `pip install openpathsampling`.

- **`import openpathsampling as paths`** — full path-sampling API (unchanged package name under `src/python/openpathsampling/`).
- **`import genai_tps`** — convenience re-exports plus **`genai_tps.backends.boltz`** for a Boltz-2–specific `DynamicsEngine` wrapper.

Boltz-2 itself is **not** vendored; keep the [`boltz/`](boltz/) submodule (or install `boltz` editable) for structure prediction. See **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** for the OpenPathSampling license and attribution.

## Install

```bash
pip install -e ".[boltz,dev]"   # torch + Boltz backend + tests
pip install -e ./boltz
# Optional — NVIDIA cuEquivariance kernels (faster Boltz pairformer; needed for boltz predict --kernels / our --kernels)
pip install -e "./boltz[cuda]"
```

Or conda: `conda env create -f environment.yml` then `conda activate genai-tps`.

## Example: TPS-style sampling on Boltz-2 diffusion

Boltz-2’s **structure module** is a diffusion sampler conditioned on trunk and pair features. This repo wires that sampler into OpenPathSampling as `BoltzDiffusionEngine` and provides a runnable script that loads a real checkpoint, rolls out one **forward diffusion trajectory** (noise → coordinates), then runs a few **shooting-style** segment resamples with path-probability logging.

**Script:** [`scripts/run_tps_boltz.py`](scripts/run_tps_boltz.py)

**Inputs:**

1. **Checkpoint** — a Boltz-2 Lightning `.ckpt` (same family you use for `boltz predict`).
2. **Conditioning bundle** — a `torch.save` / pickle `dict` with keys `s_trunk`, `s_inputs`, `feats`, and `diffusion_conditioning` (see `genai_tps.backends.boltz.utils.load_conditioning_bundle`). You typically capture these during a Boltz inference forward pass or from a saved tensor dump for your target complex.

**Run (GPU recommended):**

```bash
pip install -e ".[boltz,dev]"
pip install -e ./boltz
python scripts/run_tps_boltz.py \
  --checkpoint /path/to/boltz2.ckpt \
  --conditioning /path/to/conditioning.pt \
  --out genai_tps_boltz_run.log \
  --rounds 5
```

The script builds `BoltzSamplerCore` → `BoltzDiffusionEngine`, defines example state volumes (high-σ vs. quality proxy), generates an initial path, and appends per-round log-path-probability lines to `--out`. It is a **demonstration harness**; full TPS (e.g. `PathSimulator`, replica exchange) can be layered on the same engine and snapshots.

### Benchmark-style test (Boltz upstream example)

The pytest module [`tests/test_benchmark_prot_no_msa_tps.py`](tests/test_benchmark_prot_no_msa_tps.py) pins the TPS/path-probability stack to **Boltz’s** checked-in [`boltz/examples/prot_no_msa.yaml`](boltz/examples/prot_no_msa.yaml): a single-chain protein with `msa: empty`, as in Boltz’s own docs. It asserts the YAML sequence matches a fingerprint and runs a **mock** diffusion trajectory at that sequence length (no checkpoint). Run with:

```bash
pytest tests/test_benchmark_prot_no_msa_tps.py -m benchmark
```

### Co-folding demo (real Boltz-2, two chains)

For a **heterodimer co-folding**–style benchmark (two proteins associating), run [`scripts/run_cofolding_tps_demo.py`](scripts/run_cofolding_tps_demo.py) on [`examples/cofolding_multimer_msa_empty.yaml`](examples/cofolding_multimer_msa_empty.yaml) (Boltz multimer–style sequences with `msa: empty`). The script preprocesses the YAML like `boltz predict`, runs the **Boltz-2 trunk** once, then the **real** `AtomDiffusion` sampler: one full **diffusion-time trajectory** (noise → coordinates) and **shooting** rounds with path-probability logging. First run downloads weights into `--cache` (default `~/.boltz`). **GPU strongly recommended.** By default the script runs **without** cuEquivariance kernels so it works with a plain `pip install -e ./boltz`. For kernel acceleration (as in full `boltz predict`), install `pip install -e "./boltz[cuda]"` and add **`--kernels`**.

```bash
pip install -e ".[boltz,dev]"
pip install -e ./boltz
python scripts/run_cofolding_tps_demo.py --out ./cofolding_tps_out --diffusion-steps 32 --shoot-rounds 5
# optional: pip install -e "./boltz[cuda]"  then  ... --kernels
```

Artifacts: `trajectory_summary.json` (per-frame σ and geometry stats), `coords_trajectory.npz` (stacked coordinates), `shooting_log.txt`.

**Viewing trajectories:** [`scripts/visualize_cofolding_trajectory.py`](scripts/visualize_cofolding_trajectory.py) writes a **multi-model PDB** (one MODEL per diffusion frame) for PyMOL / ChimeraX / VMD, optional **σ/RMSD plot**, and optional **GIF** (`pip install pillow` for GIF). Example:

```bash
python scripts/visualize_cofolding_trajectory.py \
  --npz ./cofolding_tps_out/coords_trajectory.npz \
  --summary ./cofolding_tps_out/trajectory_summary.json \
  --pdb-out ./cofolding_tps_out/diffusion_traj.pdb
# PyMOL: open diffusion_traj.pdb, use state slider to scrub frames
```

## Relationship to OpenPathSampling

We ship a **snapshot** of upstream OPS inside this repo. To refresh the vendor tree, compare against a tagged upstream release and merge carefully (internal imports stay `openpathsampling.*`). Details: **[docs/vendor_openpathsampling.md](docs/vendor_openpathsampling.md)**.

## Deprecated import path

The old package name **`tps_boltz`** still works as a thin shim but emits `DeprecationWarning`. Prefer:

- `genai_tps.backends.boltz` — Boltz engine, snapshots, path probability, etc.
- `genai_tps` — top-level shortcuts (Boltz symbols + common OPS types).

## Tests

```bash
pip install -e ".[boltz,dev]"
pip install -e ./boltz   # submodule: needed for Boltz imports in backend tests
pytest tests/
```

`tests/conftest.py` adds `boltz/src` to `sys.path` when the submodule exists; you still need Boltz’s dependencies (install editable as above).

**TDD:** New behavior should land with a failing test first, then the minimal implementation. Contract tests for GPU-backed snapshots and device placement live under `tests/test_snapshot_gpu_contract.py` and `tests/test_gpu_core_device_propagation.py`. Tests marked `@pytest.mark.cuda` run only when CUDA is available.

## Documentation

- Theory note: [docs/tps_diffusion_theory.tex](docs/tps_diffusion_theory.tex) (compile with `pdflatex`).
- Vendoring policy: [docs/vendor_openpathsampling.md](docs/vendor_openpathsampling.md).

## License

genai-tps project license applies to original files; vendored OpenPathSampling remains MIT—see **THIRD_PARTY_NOTICES.md**.
