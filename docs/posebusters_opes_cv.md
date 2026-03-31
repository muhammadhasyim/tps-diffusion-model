# PoseBusters collective variables for OPES (`run_opes_tps.py`)

[PoseBusters](https://github.com/maabuu/posebusters) runs chemistry and geometry checks on predicted ligand poses relative to a protein structure (and, in **redock** modes, a reference ligand). This repo now supports two PoseBusters-backed paths in **`scripts/run_opes_tps.py`**:

- **Exact CPU PoseBusters**: upstream `posebusters==0.6.5` + RDKit + temporary PDB/SDF files.
- **Tensor-first GPU geometry subset**: project-local checks that operate directly on trajectory tensors and avoid the per-step GPU to CPU sync in the main OPES path.

## Install

```bash
pip install posebusters==0.6.5
```

Also listed in [`environment.yml`](../environment.yml) and the `enhanced_sampling` extra in [`pyproject.toml`](../pyproject.toml).

## Backend modes

Use `--posebusters-backend` to choose how PoseBusters-style checks are evaluated:

- `cpu_posebusters` — exact upstream PoseBusters. This preserves the original behavior and supports `posebusters_pass_fraction`, `posebusters_all`, and `posebusters__*`.
- `gpu_fast` — tensor-native geometric checks only. This supports `posebusters_gpu_pass_fraction`, `posebusters_gpu_all`, and `posebusters_gpu__*`.
- `hybrid` — uses the GPU subset for OPES CV values and can also run exact CPU PoseBusters every `--posebusters-cpu-fallback-every N` evaluations as a parity probe. The CPU fallback is diagnostic only; it does not replace the OPES CV value unless you explicitly use the CPU namespace.

## CV names

| Token | Meaning |
|-------|---------|
| `posebusters_pass_fraction` | Single scalar in **[0, 1]**: mean of boolean columns returned by `PoseBusters.bust` for the current frame (**cannot** be combined with other CVs in one `--bias-cv`). |
| `posebusters_all` | **Sole** `--bias-cv` value; expands internally to every default boolean check as separate dimensions (`posebusters__…`), with **one** `bust` evaluation per MC step (cached across dimensions). |
| `posebusters_gpu_pass_fraction` | Single scalar in **[0, 1]**: mean pass rate over the project-local tensor-native geometric checks listed below (**cannot** be combined with other CVs in one `--bias-cv`). |
| `posebusters_gpu_all` | **Sole** `--bias-cv` value; expands internally to every tensor-native geometry check as separate dimensions (`posebusters_gpu__…`), with **one** evaluator pass per MC step (cached across dimensions). |

Current GPU-native checks exposed by `posebusters_gpu_all`:

- `posebusters_gpu__ligand_rmsd_le_2a`
- `posebusters_gpu__ligand_pocket_dist_le_6a`
- `posebusters_gpu__ligand_contacts_ge_5`
- `posebusters_gpu__ligand_hbonds_ge_1`
- `posebusters_gpu__clash_count_le_4`
- `posebusters_gpu__ligand_max_extent_le_8a`
- `posebusters_gpu__ligand_bbox_volume_le_250a3`

To print the expanded comma-separated list for your selected backend and reference settings (same Boltz run as a full TPS job):

```bash
python scripts/run_opes_tps.py --yaml your.yaml --list-posebusters-cvs
```

You can paste that list back as `--bias-cv` only if it still matches the same mode, reference ligand, and topology.

## CLI flags

- `--posebusters-backend` — `cpu_posebusters` (exact), `gpu_fast` (tensor-native subset), or `hybrid` (GPU CVs plus reduced-cadence CPU parity probes). Default **`cpu_posebusters`**.
- `--posebusters-mode` — Preset matching upstream YAML (`dock`, `redock`, `redock_fast`, …). Default **`redock_fast`** for lighter energy checks during OPES.
- `--posebusters-reference-ligand-sdf` — Optional crystal/reference ligand for **redock***.
- `--posebusters-use-initial-ligand-ref` / `--no-posebusters-use-initial-ligand-ref` — Default **on**: build the reference ligand SDF from the **initial** trajectory frame (after `redock*` requires a reference).
- `--posebusters-ligand-chain` — NONPOLYMER chain id if several ligands exist (default: first nonpolymer chain).
- `--posebusters-cpu-fallback-every` — In `hybrid` mode, run exact upstream PoseBusters every N evaluations to check parity at reduced cadence. Default **0** (disabled).

## Requirements

- **`--topo-npz`** (or auto-detected `processed/structures/*.npz`) for Boltz topology.
- **`max_workers=0`** inside PoseBusters is enforced in code so OPES does not spawn process pools each step.
- The GPU subset does **not** require `posebusters` or RDKit for the per-step OPES evaluation path, but `hybrid` mode needs upstream PoseBusters installed for the parity probe.

## Limitations

- **Do not mix** `posebusters__*` columns with non-PoseBusters CVs in one bias.
- **Do not mix** `posebusters_gpu__*` columns with non-GPU-PoseBusters CVs in one bias.
- **Multiple ligands**: only one nonpolymer chain is scored unless you set `--posebusters-ligand-chain`.
- Column names depend on the installed PoseBusters version; keep the pin aligned with [`environment.yml`](../environment.yml).
- The GPU subset is intentionally **not** a full semantic clone of upstream PoseBusters. It only covers geometry-only checks that are practical to evaluate on tensors without RDKit chemistry perception or file-format reconstruction.

## Reference

- Paper: [Chemical Science (2024)](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d3sc04185a)
- Docs: [posebusters.readthedocs.io](https://posebusters.readthedocs.io)
