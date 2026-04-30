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

Or conda: `conda env create -f environment.yml` then `conda activate genai-tps`. The file pulls **OpenMM**, **openmmforcefields**, **OpenFF Toolkit**, **ProDy**, **PyMOL**, **pymbar**, plus editable `boltz` and `genai-tps` with `[boltz,dev,sampling]`. You do **not** need `PYTHONPATH`—editable install plus `genai_tps.subprocess_support` cover driver subprocesses.

After the first `env create`, finish setup in order (full rationale and pins live in the **`environment.yml` header**):

1. **Refresh pins on an existing env** — Avoid `conda env update` for this stack (it can leave torch in a bad state). Use **Step 1b** in `environment.yml`: a one-shot `conda install` from conda-forge for OpenMM / PLUMED / build tools.
2. **Optional PLUMED with OPES** — Conda’s `plumed` binary does not ship the `opes` module. For OPES-enabled MD, use the **`plumed2` git submodule**, pin it to the same 2.9.x line as in the env file (see comments there), then run [`scripts/build_plumed_opes.sh`](scripts/build_plumed_opes.sh) (installs into `$CONDA_PREFIX` and sets activate hooks). Skip if you do not need PLUMED OPES.
3. **PyTorch matching your driver** — With the env activated, run [`scripts/install_torch_cuda.sh`](scripts/install_torch_cuda.sh). Pass `cu121` (driver CUDA 12.0–12.7), `cu128` (12.8+ / driver ≥ 550), `cpu`, or omit the argument for **`auto`** (uses `nvidia-smi` when a GPU is present). This replaces the conda-forge torch install with the correct PyTorch wheel index and avoids broken partial reinstalls.
4. **Re-activate** — `conda deactivate && conda activate genai-tps` so **LD_LIBRARY_PATH** (torch CUDA libs) and **PLUMED_KERNEL** (if you built OPES) apply.

**Optional pip extras** (see `pyproject.toml`): `[viz]` (PyMOL), `[analysis]` (ProDy), `[sampling]` (OpenMM + pymbar + GAFF/OpenFF pins), `[cli]` (Typer + Hydra). For a **single pip line** without conda: `pip install -e ./boltz` and `pip install -e ".[boltz,dev,full]"` (`full` bundles PyMOL, ProDy, and the full OpenMM/ligand/MBAR set).

**GPU selection:** Boltz/training CLIs accept `--device cuda` or `--device cuda:N` and set the current CUDA device when PyTorch uses CUDA. OpenMM drivers accept `--openmm-device-index` (OpenMM platform property `DeviceIndex`); when omitted, scripts that tie OpenMM to Boltz infer the index from `cuda:N` (bare `cuda` is treated as device `0`). You can still use `CUDA_VISIBLE_DEVICES` to limit which GPUs a process sees.

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

For a **heterodimer co-folding**–style benchmark (two proteins associating), run [`scripts/run_cofolding_tps_demo.py`](scripts/run_cofolding_tps_demo.py) on [`examples/cofolding_multimer_msa_empty.yaml`](examples/cofolding_multimer_msa_empty.yaml) (Boltz multimer–style sequences with `msa: empty`). The script preprocesses the YAML like `boltz predict`, runs the **Boltz-2 trunk** once, then the **real** `AtomDiffusion` sampler: one full **diffusion-time trajectory** (noise → coordinates) and **shooting** rounds with path-probability logging. First run downloads weights into `--cache` (default `$BOLTZ_CACHE` or `$SCRATCH/.boltz` when those env vars are set; otherwise pass `--cache` explicitly). **GPU strongly recommended.** By default the script runs **without** cuEquivariance kernels so it works with a plain `pip install -e ./boltz`. For kernel acceleration (as in full `boltz predict`), install `pip install -e "./boltz[cuda]"` and add **`--kernels`**.

```bash
pip install -e ".[boltz,dev]"
pip install -e ./boltz
python scripts/run_cofolding_tps_demo.py --out ./artifacts/cofolding/cofolding_tps_out --diffusion-steps 32 --shoot-rounds 5
# optional: pip install -e "./boltz[cuda]"  then  ... --kernels
```

Artifacts: `trajectory_summary.json` (per-frame σ and geometry stats), `coords_trajectory.npz` (stacked coordinates), `shooting_log.txt`.

**Viewing trajectories:** [`scripts/visualize_cofolding_trajectory.py`](scripts/visualize_cofolding_trajectory.py) writes a **multi-model PDB** (one MODEL per diffusion frame) for PyMOL / ChimeraX / VMD, optional **σ/RMSD plot**, and optional **GIF** (`pip install pillow` for GIF). Example:

```bash
python scripts/visualize_cofolding_trajectory.py \
  --npz ./artifacts/cofolding/cofolding_tps_out/coords_trajectory.npz \
  --summary ./artifacts/cofolding/cofolding_tps_out/trajectory_summary.json \
  --pdb-out ./artifacts/cofolding/cofolding_tps_out/diffusion_traj.pdb
# PyMOL: open diffusion_traj.pdb, use state slider to scrub frames
```

## RLDiff-style offline RL (Boltz-2)

[`scripts/train_rl_boltz.py`](scripts/train_rl_boltz.py) implements an **offline** fine-tuning loop inspired by [RLDiff](https://github.com/oxpig/RLDiff) (DDPO-IS–style clipped surrogate and importance weights), adapted to **Boltz-2** `AtomDiffusion` stepping. Terminal rewards use **GPU-native** PoseBusters-style geometry checks in [`genai_tps/evaluation/posebusters.py`](src/python/genai_tps/evaluation/posebusters.py) (no dependency on the third-party `posebusters` package on the training path). Core helpers live under [`genai_tps/rl/`](src/python/genai_tps/rl/).

**Citation:** Broster *et al.*, *Teaching Diffusion Models Physics: Reinforcement Learning for Physically Valid Diffusion-Based Docking*, bioRxiv (2026), DOI [10.64898/2026.03.25.714128](https://doi.org/10.64898/2026.03.25.714128). See **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** for the RLDiff MIT license and attribution of derived surrogate code.

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
python -m pytest tests/
```

Use the same interpreter that installed the package (`conda activate …` or `.venv/bin/python`) so NumPy/pandas match **pyproject.toml**—a global `pytest` on system Python can pull incompatible wheels from `~/.local`.

`tests/conftest.py` adds `boltz/src` to `sys.path` when the submodule exists; you still need Boltz’s dependencies (install editable as above).

**TDD:** New behavior should land with a failing test first, then the minimal implementation. Contract tests for GPU-backed snapshots and device placement live under `tests/test_snapshot_gpu_contract.py` and `tests/test_gpu_core_device_propagation.py`. Tests marked `@pytest.mark.cuda` run only when CUDA is available.

## Documentation

- Theory note: [docs/tps_diffusion_theory.tex](docs/tps_diffusion_theory.tex) (compile with `pdflatex`).
- Vendoring policy: [docs/vendor_openpathsampling.md](docs/vendor_openpathsampling.md).

## License

genai-tps project license applies to original files; vendored OpenPathSampling remains MIT—see **THIRD_PARTY_NOTICES.md**.
