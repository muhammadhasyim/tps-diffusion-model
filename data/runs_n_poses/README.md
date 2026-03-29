# Runs N’ Poses / Zenodo data (local)

Large downloads from [Zenodo 18366081](https://zenodo.org/records/18366081) and extracted archives live here. This directory is **mostly gitignored**; only this `README.md` is tracked.

## Populate

```bash
python scripts/download_runs_n_poses_zenodo.py --out data/runs_n_poses/zenodo --preset light
python scripts/download_runs_n_poses_zenodo.py --out data/runs_n_poses/zenodo --preset medium   # + ground_truth.tar.gz
python scripts/extract_zenodo_runs_n_poses.py \
  --archive data/runs_n_poses/zenodo/ground_truth.tar.gz \
  --out data/runs_n_poses/extracted/ground_truth
```

See [docs/runs_n_poses_reproduction.md](../docs/runs_n_poses_reproduction.md) for tiers, OPES flags, and disk notes.

Suggested layout after setup:

- `zenodo/` — raw Zenodo files + `manifest.json`
- `extracted/ground_truth/` — unpacked benchmark ground truth (optional)
- `training_ligands_flat/` — optional flat `*.sdf` tree for `--skrinjar-training-ligands-dir`
