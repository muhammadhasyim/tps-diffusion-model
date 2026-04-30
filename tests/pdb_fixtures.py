"""Shared PDB text snippets for OpenMM / RMSD tests."""

from __future__ import annotations

# Minimal dipeptide (two ALA) heavy atoms — same layout as tests that need a tiny PDB.
ALA_ALA_PDB = "\n".join(
    [
        "REMARK 1 tiny test",
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C",
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C",
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O",
        "ATOM      5  CB  ALA A   1       2.000  -0.773  -1.246  1.00  0.00           C",
        "ATOM      6  N   ALA A   2       3.326   1.593   0.000  1.00  0.00           N",
        "ATOM      7  CA  ALA A   2       4.307   2.651   0.000  1.00  0.00           C",
        "ATOM      8  C   ALA A   2       5.789   2.311   0.000  1.00  0.00           C",
        "ATOM      9  O   ALA A   2       6.252   1.189   0.000  1.00  0.00           O",
        "ATOM     10  CB  ALA A   2       4.079   3.430  -1.276  1.00  0.00           C",
        "END",
        "",
    ]
)
