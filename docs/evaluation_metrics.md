# Evaluation Metrics Reference

This document describes the evaluation metrics implemented in
`genai_tps.evaluation` for assessing generative protein-ligand ensemble quality.
Metrics cover both **trajectory-level diagnostics** (unique to path sampling)
and **distribution-level benchmarks** (suitable for direct comparison with
equilibrium-ensemble models such as AnewSampling).

---

## Metric source table

| Metric | Function | Module | Source paper | When to use |
|--------|----------|--------|-------------|-------------|
| **Ligand torsion JS distance** | `torsion_js_distance` | `distribution_metrics` | Wang et al. 2026, § A.1 | Comparing ligand conformational diversity |
| **1-D Wasserstein distance** | `wasserstein_1d` | `distribution_metrics` | General / Wang et al. 2026 | Per-bond dihedral or interaction distance distributions |
| **Per-residue RMSF** | `rmsf_spearman_rmse`, `ensemble_rmsf` | `distribution_metrics` | Wang et al. 2026, § A.1 | Protein backbone flexibility |
| **Per-atom W₂ (RMWD)** | `wasserstein_2_per_atom` | `distribution_metrics` | Wang et al. 2026, Eq. 1-2 | Full 3-D per-atom positional diversity |
| **Ligand torsion JS summary** | `ligand_torsion_js_summary` | `ligand_torsions` | Wang et al. 2026, § A.1 | Per-bond torsion distribution quality |
| **Interaction fingerprint W1** | `interaction_ws_distances` | `interaction_fingerprints` | Wang et al. 2026, § A.1 | Protein-ligand contact occupancy |
| **Pairwise RMSD Pearson r** | `pairwise_rmsd_pearson` | `ensemble_atlas` | ATLAS / Vander Meersche 2024 | Internal conformational diversity |
| **PCA geometry W₂** | `pca_wasserstein2` | `ensemble_atlas` | ATLAS | Distributional shift in collective motion |
| **Transient contact Jaccard** | `transient_contact_jaccard` | `ensemble_atlas` | ATLAS | Transient structural contacts |
| **SASA exposure Pearson r** | `sasa_pearson` | `ensemble_atlas` | ATLAS | Per-residue solvent accessibility |
| **Full ATLAS benchmark** | `atlas_benchmark` | `ensemble_atlas` | ATLAS / Wang et al. 2026 | Side-by-side comparison with other methods |

---

## Distribution metrics (`distribution_metrics.py`)

### `torsion_js_distance(angles_P, angles_Q, n_bins=36)`

Computes \(\sqrt{\mathrm{JSD}(P \| Q)}\) between two dihedral angle distributions
using 36 histogram bins on \([-180°, 180°)\) and Laplace smoothing.

- **Range**: \([0, 1]\).  0 = identical; 1 = disjoint.
- **Symmetric**: \(d(P, Q) = d(Q, P)\).
- **Default bins**: 36 (10° resolution, matching AnewSampling § A.1).

### `wasserstein_1d(a, b)`

Wrapper around `scipy.stats.wasserstein_distance`.  Used for per-interaction
and per-torsion 1-D distributional comparisons.

### `rmsf_spearman_rmse(pred_rmsf, ref_rmsf)`

Returns `(Spearman ρ, RMSE)` for per-residue RMSF vectors.  High ρ (close to 1)
indicates correct relative flexibility ordering.  Low RMSE indicates accurate
absolute flexibility magnitudes.

### `wasserstein_2_per_atom(coords_P, coords_Q)`

Gaussian-approximation Wasserstein-2 per atom, decomposed as:

\[
W_2^2(\mathcal{P}_n, \mathcal{Q}_n) = \|\mu_P - \mu_Q\|^2 + \mathcal{B}(\Sigma_P, \Sigma_Q)
\]

where \(\mathcal{B}\) is the Bures metric (covariance mismatch).
Returns a dict with keys `w2_squared`, `mean_sq`, `bures`, `rmwd`.

### `ensemble_rmsf(coords, atom_mask=None)`

Per-atom \(\mathrm{RMSF}_n = \sqrt{\langle \|x_n - \langle x_n \rangle\|^2 \rangle}\).

---

## Ligand torsions (`ligand_torsions.py`)

### `enumerate_rotatable_dihedrals(mol)`

Returns all unique rotatable dihedral quartets `(i, j, k, l)` from an RDKit
molecule using the SMARTS pattern `[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]`.
Terminal single bonds (both atoms degree 1) are excluded by the `[!D1]` filter.

### `extract_dihedral_trajectory(coords, quartet)`

Computes per-frame dihedral angles in degrees using the Blondel-Karplus formula
(atan2 sign convention, range \((-180°, 180°]\)).

### `ligand_torsion_js_summary(mol, coords_pred, coords_ref, n_bins=36)`

Returns per-bond JS distances with aggregate statistics:
`mean_js`, `median_js`, `success_rate` (fraction of bonds with JS < 0.2).

---

## Interaction fingerprints (`interaction_fingerprints.py`)

Requires `prolif>=2.0` (listed under `[project.optional-dependencies] analysis`
in `pyproject.toml`; install when using interaction fingerprints).  Install with:

```bash
pip install prolif>=2.0
# or
conda install -c conda-forge prolif
```

### `compute_interaction_distances(traj, lig_sel, protein_sel)`

Runs ProLIF over an MDAnalysis universe/trajectory and returns per-interaction
per-frame binary indicator arrays.

### `interaction_ws_distances(fp_pred, fp_ref, only_shared=True)`

Computes W₁ distance per interaction label between two fingerprint dicts.
For binary indicators, W₁ = |occupation fraction difference|.

---

## ATLAS benchmark (`ensemble_atlas.py`)

### `atlas_benchmark(coords_p, coords_q, ca_mask, sasa_p, sasa_q, n_pca, contact_cutoff)`

Runs all five ATLAS metrics in one call:

| Key | Metric | Good value |
|-----|--------|-----------|
| `pairwise_rmsd_pearson` | Pearson r on pairwise RMSD lower triangles | Close to 1 |
| `pca_w2["mean_w2"]` | Mean W₂ on top-K PC projections (Å) | Close to 0 |
| `transient_contact_jaccard` | Jaccard on transient contacts | Close to 1 |
| `rmsf_pearson` | Pearson r on per-Cα RMSF | Close to 1 |
| `sasa_pearson` | Pearson r on per-residue SASA | Close to 1 |

---

## Choosing the right metric

### Path sampling diagnostics vs. ensemble quality

| Goal | Use |
|------|-----|
| Monitor OPES/TPS convergence within a run | TPS diagnostics (CVs, path length, acceptance rate) |
| Compare sampling efficiency of different OPES/TPS settings | RMSF, transient contacts, torsion JS |
| Compare genai-tps generated ensemble to MD reference | Full ATLAS benchmark + torsion JS + interaction W1 |
| Direct comparison with AnewSampling/ConfDiff | Torsion JS + RMSF Spearman/RMSE + SASA Pearson (matches their reported metrics) |
| Assess ligand pose diversity | Ligand torsion JS summary + W₂ RMWD |
| Protein backbone conformational coverage | Pairwise RMSD Pearson + PCA W₂ |

### Scientific distinction from AnewSampling

**genai-tps** generates **reaction pathways** (sequences of states connecting
two endpoints in conformation space) via Transition Path Sampling.  AnewSampling
and similar methods generate **equilibrium ensembles** without time ordering.
These are complementary:

- genai-tps excels at: kinetics (rate estimation), rare-event pathways, finding
  new binding/unbinding routes, mechanism elucidation.
- Equilibrium methods excel at: thermodynamic sampling, population ratios,
  NMR/HDX comparisons.

The distribution-level metrics in this module can be applied to genai-tps
ensembles **after** reweighting via MBAR/OPES to recover equilibrium populations.

---

## References

1. Wang Y. et al. (2026). *Learning the All-Atom Equilibrium Distribution of
   Biomolecular Interactions at Scale.* bioRxiv `10.64898/2026.03.10.710952`.
2. Vander Meersche Y. et al. (2024). *ATLAS: protein flexibility description
   from atomistic molecular dynamics simulations.* Nucleic Acids Research.
3. Xu Y., Wang Y., Luo S. et al. (2026). *Quotient-Space Diffusion Models.*
   ICLR 2026 Oral. arXiv:2604.21809.
4. Bouysset C. & Fiorucci S. (2021). *ProLIF: a library to encode molecular
   interactions as fingerprints.* J. Cheminformatics 13, 72.
