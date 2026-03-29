# Reproducing Škrinjar et al. (Runs N’ Poses) training similarity

This document ties the **[plinder-org/runs-n-poses](https://github.com/plinder-org/runs-n-poses)** submodule and Zenodo assets to full-batch similarity scoring and to **incremental** scoring for TPS trajectories in this repo.

## References

- Preprint: [bioRxiv 2025.02.03.636309](https://doi.org/10.1101/2025.02.03.636309)
- Benchmark repo (submodule): [`papers/runs-n-poses/`](../papers/runs-n-poses/)
- Zenodo concept DOI: [10.5281/zenodo.14794785](https://doi.org/10.5281/zenodo.14794785) (resolve to **latest version** record for download URLs)
- Similarity definition (paper / PLINDER benchmark): `SuCOS-pocket` = SuCOS after RDKit `rdShapeAlign.AlignMol`, multiplied by **pocket_qcov** from the PLINDER pipeline (Methods §7.4).
- **This repo’s incremental CV** (`training_sucos_pocket_qcov` in `genai_tps.analysis.skrinjar_similarity`) uses a **geometric pocket Cα proxy** and **does not** apply Foldseek’s rigid transform to the training ligand the way `papers/runs-n-poses/similarity_scoring.py` does for holo pairs — expect differences vs Zenodo `all_similarity_scores.parquet` even when RDKit matches upstream.

## Disk and RAM (order of magnitude)

| Stage | Rough size |
|-------|------------|
| wwPDB mmCIF mirror (compressed) | ~150–400 GB |
| Unpacked + PLINDER-derived trees | +0.3–1 TB typical |
| Foldseek databases (training split) | tens–low hundreds of GB |
| Zenodo `all_similarity_scores.parquet` only | ~360 MB |
| Zenodo `ground_truth.tar.gz` | ~400 MB |
| Zenodo `msa_files.tar.gz` | ~46 TB (optional; do not download unless needed) |

Plan **≥1 TB** free disk for a comfortable full PDB + DB + scratch workflow; **64–128 GB RAM** minimum for large batches, more if unchunked.

## GPU vs CPU

- **Foldseek** (release 10+): use `--gpu 1` with a recent NVIDIA driver; see [Foldseek releases](https://github.com/steineggerlab/foldseek).
- **RDKit** `rdShapeAlign`, **SuCOS-style** pharmacophore + shape terms, and **PLINDER** chemistry: **CPU** in the reference stack. The incremental module in `genai_tps.analysis.skrinjar_similarity` matches that chemistry on CPU while using GPU for Foldseek **search** only.

## 1. Download Zenodo files (no full PDB)

```bash
conda run -n genai-tps python scripts/download_runs_n_poses_zenodo.py \
  --out data/runs_n_poses/zenodo \
  --files annotations.csv,all_similarity_scores.parquet,inputs.json
```

Large archives (`ground_truth.tar.gz`, `prediction_files.tar.gz`, `msa_files.tar.gz`) are opt-in via `--files`.

## 2. OpenStructure chemical dictionary (for OpenStructure-based scoring in upstream repo)

Per [runs-n-poses #6](https://github.com/plinder-org/runs-n-poses/issues/6):

```bash
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
chemdict_tool create components.cif.gz compounds.chemlib pdb -i
export OST_COMPOUNDS_CHEMLIB="$(pwd)/compounds.chemlib"
```

(`chemdict_tool` ships with OpenStructure.)

## 3. PLINDER dataset layout

Full-batch `similarity_scoring.py` expects **`plinder`’s configured data directory** (`PLINDER_DIR` / `get_config().data.plinder_dir`) with ingested systems (see [PLINDER docs](https://github.com/plinder-org/plinder)).

Install and download data per upstream instructions, then point:

```bash
export PLINDER_DATA_DIR=/path/to/plinder/data   # or as required by your plinder version
```

## 4. PDB mirror (full reproduction)

Mirror mmCIF from wwPDB (example; check current rsync layout on [wwPDB](https://www.wwpdb.org/)):

```bash
# Example only — verify host/path in official documentation
rsync -rlpt -v -z --delete \
  rsync.ebi.ac.uk::pub/databases/rcsb/pdb-remediated/data/entries/divided/mmCIF/ \
  /path/to/pdb_mmcif/
```

The **runs-n-poses** notebooks (`input_preparation.ipynb`, PLINDER ingestion) describe how they build `new_pdb_ids.txt` and Foldseek `holo_foldseek` parquet tables before calling `similarity_scoring.py`.

## 5. Foldseek database (GPU search)

Build a Foldseek DB from a **training-cutoff** structure set (e.g. PDB IDs released before `2021-09-30` for the paper, or your Boltz-2 cutoff):

```bash
foldseek createdb training_structures/ trainingDB
# optional clustering:
# foldseek cluster trainingDB training_clu tmp ...
```

Search (GPU):

```bash
foldseek easy-search query.pdb trainingDB aln tmp --gpu 1
```

## 6. Full-batch driver (this repo)

After PLINDER + Foldseek intermediates exist (see upstream `similarity_scoring.py` and PLINDER docs):

```bash
export OST_COMPOUNDS_CHEMLIB=/path/to/compounds.chemlib
export FOLDSEEK_BIN=foldseek   # optional if on PATH
python scripts/run_skrinjar_full_similarity_batch.py --pdb-id 8cq9 --work-dir papers/runs-n-poses/scoring
```

The batch script validates environment and delegates to **`papers/runs-n-poses/similarity_scoring.py`**. It does **not** replace PLINDER ingestion or Foldseek precomputation.

## 7. Incremental scoring for TPS (`genai_tps.analysis.skrinjar_similarity`)

Use **`IncrementalSkrinjarScorer`** when you have:

- A **query** complex as PDB (e.g. Boltz last frame), and
- Either a **Foldseek** database of training structures for prefilters, and/or
- A **directory of training ligand SDFs** (curated neighbors) for ligand-only SuCOS.

**Parity note:** `geometric_pocket_qcov_ca` / `pocket_qcov_ca` superimpose receptors with Kabsch on the first `min(n_query, n_target)` Cα atoms (file order) and measure pocket overlap — it is **not** PLINDER’s pocket_qcov. Ligand–ligand scores use RDKit alignment only, **without** the Foldseek **(u, t)** rotation applied to the training ligand in the benchmark holo path.

Set **`SKRINJAR_LOG_ALIGN_FALLBACK=1`** if you need log warnings when `rdShapeAlign.AlignMol` fails and the code falls back to Crippen-only alignment. **`IncrementalSkrinjarScorer.align_mol_fallback_count`** counts those fallbacks during a run.

See module docstrings and `tests/test_skrinjar_similarity.py`.

## 8. Boltz-2 cutoff note

Zenodo README explains re-filtering `all_similarity_scores.parquet` by `target_release_date < boltz_training_cutoff`. Use the date from the **Boltz-2** paper/supplement for that model, not necessarily `2021-09-30`.
