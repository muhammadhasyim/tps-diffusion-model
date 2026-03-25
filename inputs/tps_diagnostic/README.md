# TPS Diagnostic Input Cases

Three Boltz-2 prediction inputs designed as controls for validating Transition
Path Sampling (TPS) as a diagnostic tool for co-folding diffusion models.
Each case isolates a distinct failure mode identified in the literature.

---

## Background

Two papers motivate the choice of systems:

1. **Masters, Mahmoud, Lill** (2025). "Investigating whether deep learning models
   for co-folding learn the physics of protein-ligand interactions."
   *Nature Communications* **16**, 8854.
   doi:[10.1038/s41467-025-63947-5](https://doi.org/10.1038/s41467-025-63947-5)
   — Tests adversarial binding-site mutations on AF3, Boltz-1, Chai-1, RFAA.
   Funnel metadynamics (Amber ff14SB / GAFF2 / TIP3P, 1 µs × 3 replicas) provides
   the physical ground truth.

2. **Škrinjar, Eberhardt, Durairaj, Schwede** (2025). "Have protein-ligand
   co-folding methods moved beyond memorisation?"
   *bioRxiv* 2025.02.03.636309.
   doi:[10.1101/2025.02.03.636309](https://doi.org/10.1101/2025.02.03.636309)
   — Runs N' Poses benchmark (2,600 post-cutoff systems); defines success as
   LDDT-PLI > 0.8 and ligand RMSD < 2 Å; shows steep accuracy drop in
   low-training-similarity bins for drug-like molecules.

---

## Directory layout

```
inputs/tps_diagnostic/
├── README.md                             # this file
├── case1_mek1_fzc_novel.yaml            # Case 1 -- bad/novel prediction
├── case2_cdk2_atp_wildtype.yaml         # Case 2 -- good, memorised
├── case3_cdk2_atp_packed.yaml           # Case 3 -- adversarial, confident wrong
└── reference_structures/
    ├── 1B38.cif                         # CDK2 + ATP crystal structure
    └── 7XLP.cif                         # MEK1 + FZC crystal structure
```

---

## Case 1 — Bad / novel prediction

| Attribute | Value |
|-----------|-------|
| Input file | `case1_mek1_fzc_novel.yaml` |
| System | MEK1 (MAP2K1) + FZC inhibitor |
| PDB reference | [7XLP](https://www.rcsb.org/structure/7XLP) |
| Protein residues | 382 (chain A) |
| Ligand | FZC — (1R,3S)-3-[[6-[2-chloro-4-(4-methylpyrimidin-2-yl)oxy-phenyl]-3-methyl-1H-indazol-4-yl]oxy]cyclohexan-1-amine; C25H26ClN5O2 |
| Training status | **Out of training set** for all tested co-folding models (Masters et al.) |
| Similarity metric | sucos_shape_pocket_qcov = 46.21 (low bin, Škrinjar et al.) |

**What the papers say.** Masters et al. chose 7XLP specifically because it was
not in any model's training data and has low pocket/ligand similarity to training
examples.  RFAA fails entirely; three other models achieve near-native poses only
marginally, and these are exceptions in the low-similarity regime confirmed by
Škrinjar et al.'s Figure 1B.

**Expected TPS outcome.** High variance across path-ensemble shooting moves; low
fraction of trajectories reaching state B (RMSD < 2 Å to 7XLP crystal FZC pose,
LDDT-PLI > 0.8).  Serves as a *negative control*: TPS should not find a sharp,
low-variance generative tube for genuinely novel systems.

---

## Case 2 — Good prediction, tied to training memorisation

| Attribute | Value |
|-----------|-------|
| Input file | `case2_cdk2_atp_wildtype.yaml` |
| System | CDK2 (CDKN2A) + ATP (wild-type) |
| PDB reference | [1B38](https://www.rcsb.org/structure/1B38) |
| Protein residues | 298 (chain A; N-terminal ACE cap stripped) |
| Ligands | ATP (adenosine 5′-triphosphate, CCD: ATP) + MG (magnesium ion, CCD: MG) |
| Training status | **In training data** — CDK2 + ATP is heavily represented in the PDB |

**What the papers say.** Masters et al. report AF3 achieves RMSD 0.2 Å and all
four models place ATP correctly.  Škrinjar et al. classify ATP as a *promiscuous*
cofactor (>100 training analogs in diverse pockets); success in lower similarity
bins is driven by such memorised cofactors rather than genuine generalisation
(Supplementary Figure S4–S5).

**Expected TPS outcome.** Narrow, high-weight path ensembles converging
reproducibly to state B.  Serves as a *positive control*: TPS should identify
a well-defined generative tube, reflecting the model's strong prior for this
heavily represented system.

---

## Case 3 — Adversarial: confident but physically wrong

| Attribute | Value |
|-----------|-------|
| Input file | `case3_cdk2_atp_packed.yaml` |
| System | CDK2 + ATP with 11 binding-site residues mutated to Phe |
| PDB reference | [1B38](https://www.rcsb.org/structure/1B38) (RMSD target = crystal ATP pose) |
| Protein residues | 298 (same length as WT) |
| Ligands | ATP (CCD: ATP) + MG (CCD: MG) |

**Mutations applied** (Masters et al., Methods section "Packing with bulky
hydrophobes"; all verified against PDB 1B38 chain A residue numbering):

| Position | Wild-type | Mutant | Structural context |
|----------|-----------|--------|--------------------|
| 10 | I | F | β-strand 1, N-lobe |
| 14 | T | F | Gly-rich P-loop |
| 18 | V | F | P-loop |
| 31 | A | F | β-strand 3 |
| 33 | K | F | β-strand 3, phosphate anchor (salt bridge to γ-phosphate) |
| 86 | D | F | Hinge-region linker |
| 129 | K | F | C-lobe, triphosphate contact |
| 131 | Q | F | C-lobe |
| 132 | N | F | C-lobe |
| 134 | L | F | C-lobe |
| 145 | D | F | DFG motif Asp, Mg2+ coordinator |

**Physical expectation.** Eleven Phe rings fill and pack the ATP-binding pocket,
removing all native polar contacts (salt bridges to K33, D86, K129; Mg2+
coordination via D145) and creating severe steric clash with the highly charged
triphosphate group.  Funnel metadynamics (Masters et al., Table 1) confirms the
probability of the bound state collapses to essentially zero for the packing mutant.

**What the models do.** Despite the above, Boltz-1, AF3, and Chai-1 still predict
near-native ATP placement with high ligand pLDDT and protein–ligand ipTM scores
(Masters et al., Figure 1, Supplementary Tables S1–S2).

**Expected TPS outcome.** Define two state B variants for this case:

- `B_crystal`: RMSD < 2 Å to 1B38 crystal ATP heavy atoms (what the model achieves)
- `B_phys`: physically plausible binding — no steric clashes above threshold,
  presence of favourable polar contacts

TPS path ensembles should converge to `B_crystal` but not `B_phys`, exposing
that the model's high confidence is uncalibrated — trajectories are funnelled
toward a memorised crystal geometry even when the sequence no longer supports it.
This is the sharpest of the three TPS diagnostics.

---

## How to run

### Generate structures with Boltz-2

Run each case with the MSA server (recommended) and 25 diffusion samples
(5 seeds × 5 diffusion samples per seed, matching the Škrinjar et al. protocol):

```bash
boltz predict inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
    --use_msa_server \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --out_dir outputs/tps_diagnostic/case1

boltz predict inputs/tps_diagnostic/case2_cdk2_atp_wildtype.yaml \
    --use_msa_server \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --out_dir outputs/tps_diagnostic/case2

boltz predict inputs/tps_diagnostic/case3_cdk2_atp_packed.yaml \
    --use_msa_server \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --out_dir outputs/tps_diagnostic/case3
```

To bypass the MSA server (faster, single-sequence mode, `msa: empty` already set):

```bash
boltz predict inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
    --diffusion_samples 5 \
    --sampling_steps 200 \
    --out_dir outputs/tps_diagnostic/case1
```

### Evaluate against crystal structures

Reference mmCIF files are in `reference_structures/`.

For cases 2 and 3, RMSD is computed for ATP heavy atoms aligned to 1B38
(chain B in the PDB file).  For case 1, RMSD is computed for FZC heavy atoms
aligned to 7XLP.

For case 3, also check LDDT-PLI from the Boltz-2 output and compare against
the reported pLDDT / ipTM values from Masters et al. Supplementary Tables S1–S2
to confirm the model remains overconfident after mutation.

---

## TPS state-volume mapping

| Case | State A | State B | TPS diagnostic |
|------|---------|---------|----------------|
| 1 — novel | Gaussian noise (high σ) | RMSD < 2 Å + LDDT-PLI > 0.8 | Low path weight / high variance |
| 2 — memorised WT | Gaussian noise (high σ) | RMSD < 2 Å + LDDT-PLI > 0.8 | High path weight / low variance |
| 3 — adversarial | Gaussian noise (high σ) | B_crystal and B_phys (separately) | Funnels to B_crystal but not B_phys |

See [`docs/tps_diffusion_theory.tex`](../../docs/tps_diffusion_theory.tex) for
the full mathematical formulation of state volumes, path probability, and
shooting moves in the context of Boltz-2 reverse diffusion.
