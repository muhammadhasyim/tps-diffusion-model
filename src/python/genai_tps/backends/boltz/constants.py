"""Boltz ``StructureV2`` / chain metadata constants shared across CV and analysis code.

Values match ``boltz`` ``chain_type_ids`` (PROTEIN / NONPOLYMER).  Centralised so
call sites do not drift if upstream IDs change.
"""

from __future__ import annotations

__all__ = ["NONPOLYMER_MOL_TYPE", "PROTEIN_MOL_TYPE"]

# Boltz chain_type_ids: {"PROTEIN": 0, "DNA": 1, "RNA": 2, "NONPOLYMER": 3}
PROTEIN_MOL_TYPE: int = 0
NONPOLYMER_MOL_TYPE: int = 3
