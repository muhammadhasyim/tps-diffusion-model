# Vendored Runs-N-Poses batch scoring

This directory holds a **minimal** vendored copy of the PLINDER batch similarity driver from [plinder-org/runs-n-poses](https://github.com/plinder-org/runs-n-poses), so the main repo does not depend on a git submodule for this path.

The directory `scoring/` (Foldseek parquets, PLINDER DB layout, manifests from real batch runs) is **gitignored** in the parent repo — create it locally per `docs/runs_n_poses_reproduction.md`. `scripts/run_skrinjar_full_similarity_batch.py --dry-run` does not write under `scoring/`.

## Files

| File | Purpose |
|------|---------|
| `similarity_scoring.py` | Upstream batch script (invoked with `cwd` set to this directory by `scripts/run_skrinjar_full_similarity_batch.py`). Expects `new_pdb_ids.txt`, `scoring/` layout, PLINDER data, Foldseek parquets — see `docs/runs_n_poses_reproduction.md`. |
| `LICENSE` | Apache License 2.0 as in upstream at vendor time. |

## Upstream pin

- **Repository:** https://github.com/plinder-org/runs-n-poses  
- **Commit:** `197fafc60a5cef3a9e7f8a4b0dac7c965eed3839`

Do not hand-edit `similarity_scoring.py` except for small integration patches (document the change in this file and in `THIRD_PARTY_NOTICES.md`).

## Refreshing from upstream

1. Check out upstream at the desired commit in a temporary clone.  
2. Copy `similarity_scoring.py` and `LICENSE` (if the license text changed).  
3. Update the banner at the top of `similarity_scoring.py`, this README, and `THIRD_PARTY_NOTICES.md` with the new commit SHA.

Incremental Škrinjar-style scoring used in training and campaign stage 04 lives in `genai_tps.evaluation.skrinjar_similarity` and is **not** byte-identical to this batch script; see the evaluation module docstring and `docs/runs_n_poses_reproduction.md`.
