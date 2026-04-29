# genai_tps.evaluation — TPS-oriented checks and metrics (four modules only).
#
# - tps_runner: OPES/TPS checkpointing and non-TPS ``generate_structures``.
# - posebusters: GPU-native geometry checks + upstream PoseBusters trajectory eval.
# - skrinjar_similarity: training-set similarity (Škrinjar-style).
# - terminal_ensemble_prody: ProDy PCA / normal modes on terminal PDB ensembles.
#
# Related code outside this package:
# - NPZ→PDB / topology: :mod:`genai_tps.io.boltz_npz_export`
# - CV definitions for sampling: :mod:`genai_tps.backends.boltz.collective_variables`
# - WDSM comparison plots: ``scripts/evaluation_dashboard.py``
#
# Import explicitly, e.g.
# ``from genai_tps.evaluation.skrinjar_similarity import IncrementalSkrinjarScorer``.
