# ATLAS Fine-Tuning For Boltz-2

This note describes the local ATLAS ingestion path for Boltz-2 WDSM
fine-tuning.

## Data Source

ATLAS protein-only bundles are downloaded from the DSIMB/INSERM ATLAS service.
Use the protein bundle, not the full solvated system, for the first training
path:

- API: `https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{pdb_chain}`
- Static ZIP: `https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb_chain}/{pdb_chain}_protein.zip`

ATLAS is distributed under CC-BY-NC 4.0. Confirm that this license matches the
intended use before using the dataset in a model release or commercial setting.

## Storage Strategy

Do not fetch ATLAS frames during training epochs. Download the selected bundles
once, then train from local files or a cloud/object-store mirror staged to local
NVMe.

Recommended layout:

- `data/atlas/ids/*.txt`: curated ID lists and splits.
- `data/atlas/raw/{atlas_id}/{atlas_id}_protein.zip`: immutable downloaded ZIPs.
- `data/atlas/extracted/{atlas_id}/`: rebuildable extraction cache.
- `data/atlas/processed/{run}/{atlas_id}/training_dataset.npz`: WDSM arrays.
- `data/atlas/processed/{run}/{atlas_id}/dataset_split/`: optional train/val split.
- `data/atlas/manifests/*.json`: download and conversion provenance.

## Important Limitation

The current WDSM trainer builds one Boltz-2 inference bundle from one input YAML
and checks that the training coordinates have the same atom count/order. That
means this path supports fine-tuning **per ATLAS target/YAML**. A mixed
multi-protein ATLAS trainer would require a larger data-loader change so each
batch item carries its own Boltz conditioning features.

## Pilot Workflow

Create `data/atlas/ids/pilot_ids.txt`:

```text
16pk_A
```

Download protein bundles:

```bash
python scripts/atlas/00_download_atlas.py \
  --ids-file data/atlas/ids/pilot_ids.txt \
  --raw-dir data/atlas/raw \
  --manifest data/atlas/manifests/pilot_download.json \
  --workers 4
```

Prepare Boltz-aligned WDSM NPZ files. Provide either a topology NPZ directory or
one Boltz YAML per ATLAS ID. YAML filenames should match `{atlas_id}.yaml`.

```bash
python scripts/atlas/01_prepare_atlas_wdsm.py \
  --ids-file data/atlas/ids/pilot_ids.txt \
  --raw-dir data/atlas/raw \
  --extract-dir data/atlas/extracted \
  --processed-dir data/atlas/processed/pilot \
  --yaml-dir data/atlas/yamls \
  --cache ~/.boltz \
  --stride 20 \
  --max-frames-per-replicate 256 \
  --val-fraction 0.1
```

Fine-tune one prepared target:

```bash
python scripts/atlas/02_finetune_boltz2_atlas.py \
  --yaml data/atlas/yamls/16pk_A.yaml \
  --data data/atlas/processed/pilot/16pk_A/dataset_split/train.npz \
  --val-data data/atlas/processed/pilot/16pk_A/dataset_split/val.npz \
  --out outputs/atlas_finetune/pilot \
  --loss-types true-quotient \
  --epochs 10 \
  --batch-size 4 \
  --device cuda
```

For multiple IDs, use `--ids-file`, `--prepared-dir`, and a JSON `--yaml-map`
that maps ATLAS IDs to YAML paths. The script will run independent per-target
fine-tunes under the output directory.

## Scientific Defaults

- ATLAS frames are ordinary MD samples, so prepared datasets use `logw = 0.0`.
- Start with stride or max-frame subsampling; full ATLAS trajectories are too
  large for pilot work.
- Validate with held-out proteins when making claims about generalization. A
  random frame split from the same trajectory is useful for smoke testing, but
  not enough for a scientific generalization claim.
