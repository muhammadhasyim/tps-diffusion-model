"""Unit tests for Škrinjar-style incremental similarity (RDKit + geometry + optional Foldseek)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import genai_tps.evaluation.skrinjar_similarity as sk


MINI_COMPLEX_PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG X   1       1.900   2.500   0.000  1.00 20.00           C
HETATM    5  C2  LIG X   1       2.500   3.200   0.000  1.00 20.00           C
CONECT    4    5
END
"""

# Same coordinates; alternate PDB atom-name column padding (common in the wild).
MINI_COMPLEX_PDB_CA_LEFT = """\
ATOM      1 CA   ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2 CA   ALA A   2       3.800   0.000   0.000  1.00 20.00           C
ATOM      3 CA   ALA A   3       7.600   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG X   1       1.900   2.500   0.000  1.00 20.00           C
HETATM    5  C2  LIG X   1       2.500   3.200   0.000  1.00 20.00           C
CONECT    4    5
END
"""

PROTEIN_BACKBONE_PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00 20.00           C
"""


def _write_query_pdb_with_ligand(path: Path, ligand_mol) -> None:
    """Minimal protein backbone + RDKit ligand coordinates for scoring tests."""
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415 — MolToPDBBlock

    lb = Chem.MolToPDBBlock(ligand_mol)
    lig_lines = [ln for ln in lb.splitlines() if ln.startswith(("ATOM", "HETATM", "CONECT"))]
    path.write_text(PROTEIN_BACKBONE_PDB + "\n".join(lig_lines) + "\nEND\n", encoding="utf-8")


def test_pocket_qcov_identical(tmp_path: Path) -> None:
    q = tmp_path / "q.pdb"
    t = tmp_path / "t.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    t.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    qc = sk.pocket_qcov_ca(q, t, pocket_radius=6.0, match_cutoff=2.0)
    assert qc == pytest.approx(1.0)
    assert sk.geometric_pocket_qcov_ca(q, t, pocket_radius=6.0, match_cutoff=2.0) == pytest.approx(
        qc
    )


def test_ca_columns_left_padded_parse_matches_standard(tmp_path: Path) -> None:
    std = tmp_path / "std.pdb"
    left = tmp_path / "left.pdb"
    std.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    left.write_text(MINI_COMPLEX_PDB_CA_LEFT, encoding="utf-8")
    n_std = len(sk.ca_coords_from_pdb(std))
    n_left = len(sk.ca_coords_from_pdb(left))
    assert n_left == n_std == 3


def test_kabsch_rotation_maps_pocket(tmp_path: Path) -> None:
    q = tmp_path / "q.pdb"
    t = tmp_path / "t.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    lines = MINI_COMPLEX_PDB.splitlines()
    shifted = []
    for line in lines:
        if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) >= 54:
            x = 100.0 + float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            shifted.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        else:
            shifted.append(line)
    t.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    qc = sk.pocket_qcov_ca(q, t, pocket_radius=6.0, match_cutoff=2.5)
    assert qc == pytest.approx(1.0)


def test_sucos_identical_embedded_molecules() -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=0)
    m2 = Chem.Mol(m)
    s = sk.sucos_shape_after_align(m, m2)
    assert s >= 0.85


def test_default_conf_id_zero_when_conformer_present() -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    m = Chem.MolFromSmiles("N")
    AllChem.EmbedMolecule(m, randomSeed=0)
    assert m.GetNumConformers() >= 1
    assert sk._default_conf_id(m) == 0


def test_alignmol_fallback_warns_and_counts(monkeypatch, caplog, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    monkeypatch.setenv("SKRINJAR_LOG_ALIGN_FALLBACK", "1")
    caplog.set_level(logging.WARNING)
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=0)
    m2 = Chem.Mol(m)
    with patch("rdkit.Chem.rdShapeAlign.AlignMol", side_effect=RuntimeError("forced")):
        sk.align_molecules(m, m2)
    assert "AlignMol failed" in caplog.text or "Crippen-only" in caplog.text

    phenol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(phenol, randomSeed=1)
    sdf = tmp_path / "t.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(phenol)
    w.close()
    q = tmp_path / "q.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    scorer = sk.IncrementalSkrinjarScorer(
        training_ligand_sdfs=(sdf,),
        enable_coord_cache=False,
    )
    before = scorer.align_mol_fallback_count
    with patch("rdkit.Chem.rdShapeAlign.AlignMol", side_effect=RuntimeError("forced")):
        v = scorer.score_pdb_file(q)
    assert 0.0 <= v <= 1.0
    assert scorer.align_mol_fallback_count > before


def test_parse_foldseek_style_line() -> None:
    text = "query.pdb\t1abc_A\t0.5\t500\n"
    ids = sk.parse_pdb_ids_from_foldseek_output(text, max_hits=5)
    assert "1abc" in ids


def test_parse_m8_blast_style_subject_column() -> None:
    line = "q\t8abc_A\t25.0\t50\t...\n"
    ids = sk.parse_pdb_ids_from_foldseek_output(line, max_hits=5)
    assert ids == ["8abc"]


def test_parse_pdb_id_from_subject_variants() -> None:
    assert sk._pdb_id_from_subject_field("9xyz_chainA") == "9xyz"
    assert sk._pdb_id_from_subject_field("prefix_1abc_suffix") == "1abc"


@pytest.mark.skipif(not shutil.which("foldseek"), reason="foldseek not on PATH")
def test_foldseek_easy_search_smoke(tmp_path: Path) -> None:
    q = tmp_path / "q.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    subprocess.run(
        ["foldseek", "createdb", str(q), str(tmp_path / "qdb")],
        check=True,
        capture_output=True,
        text=True,
    )
    out = sk.foldseek_easy_search(
        q,
        tmp_path / "qdb",
        use_gpu=False,
        extra_args=["--max-seqs", "1"],
    )
    assert isinstance(out, str)


def test_mol_from_sdf_path_sanitized_for_featmaps(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    sdf = tmp_path / "benz.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(mol)
    w.close()
    loaded = sk.mol_from_sdf_path(sdf)
    assert loaded is not None
    assert loaded.GetNumHeavyAtoms() == 6
    # Unsanitized MolFromMolFile(sanitize=False) + FeatMaps used to yield 0.0 / C++ abort.
    fm = sk.get_feature_map_score(loaded, loaded.__copy__())
    assert fm > 0.0


def test_incremental_scorer_ligand_sdf(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    sdf = tmp_path / "train.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(m)
    w.close()

    q = tmp_path / "q.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")

    scorer = sk.IncrementalSkrinjarScorer(
        training_ligand_sdfs=(sdf,),
        enable_coord_cache=False,
    )
    val = scorer.score_pdb_file(q)
    assert 0.0 <= val <= 1.0


def test_incremental_scorer_similar_ligand_beats_dissimilar(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    # Query: fixed mini complex (aliphatic 2C HETATM); training SDF must be sanitized
    # (see mol_from_sdf_path) or FeatMaps returns 0 for all pairs.
    q = tmp_path / "q.pdb"
    q.write_text(MINI_COMPLEX_PDB, encoding="utf-8")
    q_mol = sk.ligand_mol_from_pdb(q)
    assert q_mol is not None

    # Same ligand geometry as query vs unrelated aromatic — stable ordering for SuCOS.
    sdf_closer = tmp_path / "same_as_query.sdf"
    w = Chem.SDWriter(str(sdf_closer))
    w.write(q_mol)
    w.close()

    aromatic = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(aromatic, randomSeed=22)
    sdf_far = tmp_path / "benzene.sdf"
    w = Chem.SDWriter(str(sdf_far))
    w.write(aromatic)
    w.close()

    hi = sk.IncrementalSkrinjarScorer(
        training_ligand_sdfs=(sdf_closer,),
        enable_coord_cache=False,
    ).score_pdb_file(q)
    lo = sk.IncrementalSkrinjarScorer(
        training_ligand_sdfs=(sdf_far,),
        enable_coord_cache=False,
    ).score_pdb_file(q)
    assert hi > lo
    assert hi > 0.15


def test_score_coords_with_topo_npz_if_present(tmp_path: Path) -> None:
    pytest.importorskip("boltz")
    from genai_tps.io.boltz_npz_export import load_topo  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("**/processed/structures/*.npz"))
    if not candidates:
        pytest.skip("No Boltz processed/structures/*.npz under repository checkout")
    topo_npz = candidates[0]
    structure, n_struct = load_topo(topo_npz)
    coords = np.asarray(structure.atoms["coords"], dtype=np.float32)[: int(n_struct)]

    scorer = sk.IncrementalSkrinjarScorer(
        training_ligand_sdfs=(),
        scratch_dir=tmp_path / "sk",
        enable_coord_cache=False,
    )
    val = scorer.score_coords(coords, structure, int(n_struct))
    if np.isnan(val):
        pass
    else:
        assert -1e-6 <= val <= 1.0 + 1e-6


def test_load_parquet_similarity_optional(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq  # noqa: PLC0415

    p = tmp_path / "t.parquet"
    table = pa.table({"sucos_shape_pocket_qcov": [0.42, 0.99]})
    pq.write_table(table, p)
    v = sk.load_parquet_similarity_column(p, "sucos_shape_pocket_qcov", row_index=0)
    assert v == pytest.approx(0.42)


def test_download_zenodo_manifest_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "download_runs_n_poses_zenodo.py"
    out = tmp_path / "zen"
    r = subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--all-metadata-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    man = out / "manifest.json"
    assert man.is_file()
    assert "files" in man.read_text()


def test_batch_wrapper_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_skrinjar_full_similarity_batch.py"
    rnp = root / "papers" / "runs-n-poses" / "similarity_scoring.py"
    if not rnp.is_file():
        pytest.skip("runs-n-poses submodule not checked out")
    r = subprocess.run(
        [sys.executable, str(script), "--dry-run", "8cq9"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert r.returncode == 0, r.stderr


def test_opes_bias_cv_name_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_opes_tps.py").read_text(encoding="utf-8")
    assert '"training_sucos_pocket_qcov"' in text
    assert "_SINGLE_CV_NAMES" in text


def test_regression_parquet_row_if_present() -> None:
    root = Path(__file__).resolve().parents[1]
    pq_path = root / "data" / "runs_n_poses" / "zenodo" / "all_similarity_scores.parquet"
    if not pq_path.is_file():
        pytest.skip("Zenodo parquet not present under data/runs_n_poses/zenodo/")
    pa_pq = pytest.importorskip("pyarrow.parquet")

    table = pa_pq.read_table(pq_path)
    names = table.column_names
    col = "sucos_shape_pocket_qcov" if "sucos_shape_pocket_qcov" in names else None
    if col is None:
        for c in names:
            if "sucos" in c.lower() and "pocket" in c.lower():
                col = c
                break
    if col is None:
        pytest.skip("No sucos/pocket column in parquet")
    v0 = float(table.column(col)[0].as_py())
    v_read = sk.load_parquet_similarity_column(pq_path, col, row_index=0)
    assert v_read is not None
    assert v_read == pytest.approx(v0, rel=1e-9, abs=1e-9)
    if table.num_rows > 1:
        v1 = float(table.column(col)[1].as_py())
        v_read1 = sk.load_parquet_similarity_column(pq_path, col, row_index=1)
        assert v_read1 is not None
        assert v_read1 == pytest.approx(v1, rel=1e-9, abs=1e-9)
