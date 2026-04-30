# Third-party notices

## OpenPathSampling

This repository includes a **vendored copy** of [OpenPathSampling](https://github.com/openpathsampling/openpathsampling) under `src/python/openpathsampling/`, for path sampling (TPS, TIS, etc.) as the driver layer for generative trajectory analysis.

- **Upstream:** https://github.com/openpathsampling/openpathsampling  
- **License:** MIT License  
- **Upstream license file:** `src/python/openpathsampling/LICENSE.openpathsampling-upstream` (copy of upstream `LICENSE` at vendor time)

Copyright and permission notices from the MIT license apply to that subtree. The genai-tps project may modify vendored files for integration; track upstream releases when updating the vendor tree.

## Runs-N-Poses (vendored batch similarity script)

A **vendored copy** of the PLINDER full-batch driver `similarity_scoring.py` from [plinder-org/runs-n-poses](https://github.com/plinder-org/runs-n-poses) lives under `third_party/runs_n_poses/`.

- **Upstream:** https://github.com/plinder-org/runs-n-poses  
- **Pinned commit** (at vendor time): `197fafc60a5cef3a9e7f8a4b0dac7c965eed3839`  
- **License:** Apache License 2.0 — see `third_party/runs_n_poses/LICENSE` (includes Schwede Lab copyright notice from upstream).  
- **Vendor notes:** `third_party/runs_n_poses/README.vendor.md`

The batch script is invoked by `scripts/run_skrinjar_full_similarity_batch.py` with `cwd` set to that directory; it remains dependent on PLINDER, Foldseek intermediates, and `new_pdb_ids.txt` as documented in `docs/runs_n_poses_reproduction.md`.

**Incremental** Škrinjar-style scoring for TPS lives in `genai_tps.evaluation.skrinjar_similarity` and is **adapted** (geometric pocket proxy, no Foldseek **(u, t)** on the training ligand in that path); it is not byte-identical to the vendored batch file. Document any intentional edits to `third_party/runs_n_poses/similarity_scoring.py` in `README.vendor.md` and in this file.

## RLDiff (reference submodule and derived training code)

The repository may include **[RLDiff](https://github.com/oxpig/RLDiff)** as a **git submodule** under `RLDiff/` (MIT License, University of Oxford). The paper *Teaching Diffusion Models Physics: Reinforcement Learning for Physically Valid Diffusion-Based Docking* (bioRxiv, DOI [10.64898/2026.03.25.714128](https://doi.org/10.64898/2026.03.25.714128)) describes that framework.

A small portion of the offline **PPO-style clipped surrogate** in `src/python/genai_tps/rl/ppo_surrogate.py` is **derived from** RLDiff’s `utils/train_utils.py` (`compute_loss` and reward-normalization patterns), adapted for Boltz-2. The MIT license and copyright notice from `RLDiff/LICENSE` apply to those derived parts. The submodule is provided for attribution and comparison; runtime training uses `genai_tps.rl` and does not import DiffDock or the upstream `posebusters` package on the hot path.

## PLUMED 2 (git submodule, fork with PRINT patch)

The **`plumed2/`** directory is a **git submodule** used to build a PLUMED kernel with the **opes** module for OpenMM (`scripts/build_plumed_opes.sh`). The submodule URL points at **[muhammadhasyim/plumed2](https://github.com/muhammadhasyim/plumed2)**, a fork of [plumed/plumed2](https://github.com/plumed/plumed2), branch **`tps/v2.9.2-print-heavy-flush`** (based on tag `v2.9.2`). That branch adds an optional **`HEAVY_FLUSH`** flag to the **`PRINT`** action so COLVAR-style outputs reopen the file path after each line (mitigates stale descriptors if the path was renamed on disk).

- **Upstream:** https://github.com/plumed/plumed2 (LGPL v3)  
- **Fork:** https://github.com/muhammadhasyim/plumed2  
- **License:** same as upstream PLUMED (see `plumed2/COPYING`)

Use `--plumed-colvar-heavy-flush` with `run_openmm_opes_md.py` / Stage 00 only after installing PLUMED built from this submodule branch; stock conda PLUMED will reject the unknown keyword.
