# OneOPES-style replica exchange on OpenMM + PLUMED (design note)

This document records **Option B** from the replica-exchange plan: explicit
multi-replica exchange combining **OPES_METAD_EXPLORE**, auxiliary CV biases,
and **OPES_EXPANDED** / multithermal layers, as in Rizzi *et al.* (J. Chem.
Theory Comput. **19**, 5731–5742, 2023) and the PLUMED-NEST project
[plumID:23.011](https://www.plumed-nest.org/eggs/23.011/). The reference
implementation uses **GROMACS** `-replex` with per-replica `plumed.dat` files;
this repo’s production path is **OpenMM** + `openmm-plumed`, so a port requires
new orchestration described below.

## Reference OneOPES stratification (GROMACS)

- **Replica 0:** single `OPES_METAD_EXPLORE` on leading CVs; convergence-focused;
  reweighting uses this replica’s trajectory.
- **Replicas 1–3:** same leading bias plus progressively more **auxiliary**
  `OPES_METAD_EXPLORE` actions on extra CVs (weaker `BARRIER`, slower `PACE`).
- **Replicas 4–7:** same as replica 3 plus **`ECV_MULTITHERMAL`** +
  **`OPES_EXPANDED`** on the potential energy (or `E + pV` under NPT).

Replicas exchange configurations; each replica maintains **its own** kernel
history (not `WALKERS_MPI` shared bias).

## Auxiliary CVs for ligand–protein systems

The alanine dipeptide example uses backbone torsions and selected distances.
For Boltz-aligned protein–ligand targets in this repo, candidate auxiliary
sets (to be validated per system):

- Additional **torsions** (ligand or protein–ligand χ angles).
- **Minimum distances** between ligand functional groups and defined pocket
  atoms (already partially covered by `lig_dist` / `lig_contacts`).
- **Solvation** or **coordination** variants not collinear with the leading 3D
  CV set.

Auxiliary biases must use **small `BARRIER`** and **slow `PACE`** relative to
the leading OPES so they perturb the Hamiltonian gently.

## OpenMM architecture options

### A. `openmmtools.multistate.ReplicaExchangeSampler`

- **Pros:** battle-tested exchange statistics, storage, neighbor / swap-all
  policies.
- **Cons:** each thermodynamic state is normally a full `System`; here the
  **same** chemistry is used and only **PLUMED scripts** differ per replica.
  You likely need one `System` clone per replica with a different
  `PlumedForce(script_i)` attached, all at the same temperature, and custom
  logic so “thermodynamic state *i*” maps to `script_i`. Verify that
  `ReplicaExchangeSampler` allows identical temperature with different
  Hamiltonians (Hamiltonian replica exchange pattern).

### B. Lightweight custom driver

- Maintain `N` `openmm.app.Simulation` instances (or one context per replica
  if memory is tight and you serialize).
- Each step: advance all replicas `n_steps` MD; compute total potential +
  biases for exchange Metropolis between pairs `i`, `j`.
- **Critical:** exchange acceptance needs energy of configuration **A**
  evaluated in replica **B**’s biases. With `PlumedForce`, that implies either
  (1) temporary swap of `PlumedForce` / script on a scratch context, or (2)
  PLUMED `driver`-style energy evaluation (not exposed in `openmm-plumed` by
  default). Prototype: duplicate `System` with only replica `j`’s
  `PlumedForce`, set positions from replica `i`, query bias energy from force
  group.

### C. File-based “replicas” (no exchange)

- Run `N` independent jobs; merge `COLVAR` for analysis. This is **Option C**
  only; it does not implement OneOPES.

## Per-replica PLUMED state

- **KERNELS / STATE:** separate files per replica (e.g. `KERNELS.rep3`,
  `STATE.rep3`) or subdirectories `rep003/` under `opes_states/`.
- **OPES_EXPANDED:** separate `OPES_EXPANDED_DELTAFS` and `STATE_EXPANDED`
  per replica when multithermal is enabled.
- **Restart:** resume must load matching pairs `(STATE, STATE_EXPANDED)` for
  each replica that used expanded sampling.

## MPI and `WALKERS_MPI`

The vendored `build_plumed_opes.sh` build uses `--enable-mpi=no`, so PLUMED
`WALKERS_MPI` is **not** available. OneOPES in the literature does **not**
require MPI for exchange (GROMACS handles replica communication). An OpenMM
port likewise does **not** need an MPI-enabled PLUMED unless you later add
shared-bias multiple walkers.

## Suggested implementation order

1. **Option A (done):** single-replica `OPES_EXPANDED` + `ECV_MULTITHERMAL`
   alongside `OPES_METAD` / `OPES_METAD_EXPLORE` in `plumed_opes.py`.
2. Prototype **two-replica** Hamiltonian exchange in a throwaway script: same
   `System`, two `PlumedForce` variants, validate Metropolis acceptance against
   a toy system.
3. Scale to **8 replicas** with OneOPES stratified `plumed.dat` templates
   generated from a small Python table (mirror
   `Alanine/OneOPES_MultiCV/{0..7}/` in the OneOPES GitHub archive).
4. Integrate into `openmm_md_runner` or a dedicated
   `openmm_oneopes_repex.py` driver to avoid complicating the single-replica
   campaign path.

## References

- Invernizzi, Piaggi, Parrinello, *Phys. Rev. X* **10**, 041034 (2020) — unified
  OPES target distributions (CV + expanded).
- Rizzi *et al.*, *J. Chem. Theory Comput.* **19**, 5731–5742 (2023) — OneOPES.
- PLUMED Masterclass 22.3 — OPES tutorial.
- PLUMED-NEST plumID:23.011 — input files and analysis scripts.
