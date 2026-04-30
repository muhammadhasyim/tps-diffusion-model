"""Run PLUMED's OPES ``FES_from_Reweighting.py`` on a COLVAR file (subprocess).

The implementation lives in the vendored PLUMED tree
``plumed2/user-doc/tutorials/others/opes-metad/FES_from_Reweighting.py``
(GPL-2.0 / LGPL per PLUMED).  This module only locates that script and invokes
it with ``subprocess`` so we do not fork the reweighting mathematics into
``genai_tps``.

Typical Stage-0 layout::

    <case>/openmm_opes_md/opes_states/COLVAR

Output is written next to ``COLVAR`` (same directory as *outfile* basename
when *cwd* is set to ``colvar_path.parent``).

**Why a reweighted FES can look smooth while ``KERNELS`` looks busy**

PLUMED OPES stores a **compressed** kernel representation: the ``KERNELS``
history can list many deposition events, but ``opes.nker`` in ``COLVAR`` is the
**current** number of merged kernels in CV space. Reweighting then applies a
**KDE** (or PLUMED's equivalent script) with finite bandwidth, so the estimated
free energy is a **smooth** functional of the biased histogram, not a raw sum
of spiky Gaussians. That is expected behaviour, not a sign that the bias was
ignored.  For OneOPES (``pp.proj``, ``cmap``), when the user does not pass two
``sigma`` values, :func:`reweighting_kwargs_from_colvar_path` defaults the KDE
bandwidth to the **mean kernel widths** read from ``KERNELS`` (falling back to
``0.09,0.16``) so the ``.dat`` is not grossly over-smoothed relative to OPES.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from genai_tps.simulation.fes_io import load_plumed_kernels_2d
from genai_tps.subprocess_support import repository_root


def read_colvar_field_names(colvar_path: Path) -> list[str]:
    """Return column names from the first ``#! FIELDS`` line in a PLUMED COLVAR."""
    colvar_path = colvar_path.expanduser().resolve()
    with colvar_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#! FIELDS"):
                return s.split()[2:]
    raise ValueError(f"No #! FIELDS line in {colvar_path}")


def _default_sigma_2d_from_kernels_mean(kernels_path: Path) -> str:
    """Mean per-CV Gaussian widths from PLUMED 2D ``KERNELS`` for reweighting defaults.

    When the user does not pass exactly two ``sigma`` values, OneOPES reweighting
    should use a bandwidth comparable to OPES kernel sigmas — not a much larger
    placeholder (e.g. ``0.3,0.5``), which over-smooths the FES ``.dat``.
    """
    if not kernels_path.is_file():
        return "0.09,0.16"
    try:
        _, _, sigmas, _ = load_plumed_kernels_2d(kernels_path)
        if sigmas.size == 0:
            return "0.09,0.16"
        s0 = float(np.mean(sigmas[:, 0]))
        s1 = float(np.mean(sigmas[:, 1]))
        if not (np.isfinite(s0) and np.isfinite(s1) and s0 > 0.0 and s1 > 0.0):
            return "0.09,0.16"
        return f"{s0:.10g},{s1:.10g}"
    except (OSError, ValueError, IndexError):
        return "0.09,0.16"


def reweighting_kwargs_from_colvar_path(
    colvar_path: Path,
    *,
    sigma_arg: str,
) -> dict[str, object]:
    """Infer ``run_fes_from_reweighting_script`` kwargs from COLVAR ``FIELDS``.

    Recognizes OneOPES-style columns (``pp.proj`` + ``cmap``), 3D ligand–pocket
    OPES (``lig_contacts``), and the default 2D pair (``lig_rmsd``, ``lig_dist``).
    """
    colvar_path = colvar_path.expanduser().resolve()
    field_names = read_colvar_field_names(colvar_path)
    parent = colvar_path.parent
    n_sig = len([p for p in sigma_arg.split(",") if p.strip()])
    if "pp.proj" in field_names and "cmap" in field_names:
        sigma_use = (
            sigma_arg
            if n_sig == 2
            else _default_sigma_2d_from_kernels_mean(parent / "KERNELS")
        )
        return {
            "cv_names": "pp.proj,cmap",
            "sigma": sigma_use,
            "grid_bin": "100,100",
            "outfile": parent / "fes_reweighted_2d.dat",
        }
    if "lig_contacts" in field_names:
        sigma_use = sigma_arg if n_sig == 3 else "0.3,0.5,1.0"
        return {
            "cv_names": "lig_rmsd,lig_dist,lig_contacts",
            "sigma": sigma_use,
            "grid_bin": "40,40,40",
            "outfile": parent / "fes_reweighted_3d.dat",
        }
    return {
        "cv_names": "lig_rmsd,lig_dist",
        "sigma": sigma_arg,
        "grid_bin": "100,100",
        "outfile": parent / "fes_reweighted_2d.dat",
    }


def fes_from_reweighting_script_path(repo_root: Path | None = None) -> Path:
    """Return path to PLUMED tutorial ``FES_from_Reweighting.py``."""
    root = repo_root if repo_root is not None else repository_root()
    return (
        root
        / "plumed2"
        / "user-doc"
        / "tutorials"
        / "others"
        / "opes-metad"
        / "FES_from_Reweighting.py"
    )


def run_fes_from_reweighting_script(
    *,
    colvar_path: Path,
    outfile: Path,
    temperature_k: float,
    sigma: str,
    cv_names: str = "lig_rmsd,lig_dist",
    bias_name: str = "opes.bias",
    grid_min: str | None = None,
    grid_max: str | None = None,
    grid_bin: str = "100,100",
    skiprows: int = 0,
    blocks: int = 1,
    repo_root: Path | None = None,
) -> None:
    """Run PLUMED's reweighted KDE FES script on *colvar_path*.

    Parameters
    ----------
    colvar_path
        Path to PLUMED ``COLVAR`` (must exist; first line must contain
        ``FIELDS`` per PLUMED convention).
    outfile
        Destination path.  The script is run with ``cwd=colvar_path.parent``;
        pass ``outfile`` under that directory so ``-o`` uses a plain filename.
    temperature_k
        Simulation temperature in Kelvin (passed as ``--temp``; kBT in kJ/mol
        inside the PLUMED script uses 0.0083144621 × T).
    sigma
        KDE bandwidth(s), comma-separated (e.g. ``\"0.3,0.5\"`` for 2D OPES,
        or three values for 3D).
    cv_names
        Comma-separated CV labels matching ``FIELDS`` names.  Three CVs use an
        internal NumPy implementation (same KDE idea as the PLUMED script);
        one or two CVs still invoke ``FES_from_Reweighting.py``.
    bias_name
        Bias column name (e.g. ``opes.bias``).
    grid_min, grid_max
        If set, forwarded as ``--min`` / ``--max`` (comma-separated per CV for
        2D).  If ``None``, the PLUMED script auto-fits bounds from the data.
    grid_bin
        Comma-separated bin counts (``100,100`` for 2D; three integers for 3D).
    skiprows
        Rows to skip after the header (burn-in).
    blocks
        Block averaging for uncertainty (``1`` = single estimate).
    repo_root
        Repository root containing ``plumed2/``.  Default: auto-detect.

    Raises
    ------
    FileNotFoundError
        If the tutorial script or ``colvar_path`` is missing.
    subprocess.CalledProcessError
        If the script exits non-zero.
    """
    colvar_path = colvar_path.expanduser().resolve()
    outfile = outfile.expanduser().resolve()
    if not colvar_path.is_file():
        raise FileNotFoundError(f"COLVAR not found: {colvar_path}")
    if colvar_path.stat().st_size < 32:
        raise ValueError(f"COLVAR is empty or too small to contain FIELDS: {colvar_path}")
    with colvar_path.open(encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    if "FIELDS" not in head:
        raise ValueError(
            f"COLVAR first line must contain PLUMED FIELDS marker: {colvar_path} "
            f"(got {head[:120]!r})"
        )
    cwd = colvar_path.parent
    if outfile.parent.resolve() != cwd.resolve():
        raise ValueError(
            f"outfile parent must match COLVAR directory ({cwd}); got {outfile.parent}"
        )

    cv_tokens = [c.strip() for c in cv_names.split(",") if c.strip()]
    if len(cv_tokens) == 3:
        if int(blocks) != 1:
            raise ValueError(
                "Three CV reweighted FES uses the internal 3D KDE path; "
                "use --blocks 1 (block uncertainty not implemented for 3D)."
            )
        sig_parts = [float(x.strip()) for x in sigma.split(",") if x.strip()]
        if len(sig_parts) != 3:
            raise ValueError(
                f"Three CVs require three comma-separated --sigma values; got {sigma!r}"
            )
        bin_parts = [int(float(x.strip())) for x in grid_bin.split(",") if x.strip()]
        if len(bin_parts) != 3:
            raise ValueError(
                f"Three CVs require three comma-separated --bin values; got {grid_bin!r}"
            )

        def _triple_or_none(
            raw: str | None,
        ) -> tuple[float, float, float] | None:
            if raw is None:
                return None
            parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if len(parts) != 3:
                raise ValueError(
                    f"Expected three comma-separated floats for grid bounds; got {raw!r}"
                )
            return (parts[0], parts[1], parts[2])

        from genai_tps.simulation.reweighted_fes_kde import write_reweighted_fes_3d

        write_reweighted_fes_3d(
            colvar_path,
            outfile,
            temperature_k=float(temperature_k),
            sigma=(sig_parts[0], sig_parts[1], sig_parts[2]),
            cv_names=(cv_tokens[0], cv_tokens[1], cv_tokens[2]),
            bias_name=bias_name,
            grid_bin=(bin_parts[0], bin_parts[1], bin_parts[2]),
            grid_min=_triple_or_none(grid_min),
            grid_max=_triple_or_none(grid_max),
            skiprows=int(skiprows),
        )
        return

    script = fes_from_reweighting_script_path(repo_root)
    if not script.is_file():
        raise FileNotFoundError(
            f"PLUMED tutorial script not found: {script}. "
            "Initialize the plumed2 submodule or clone PLUMED with tutorials."
        )

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--colvar",
        colvar_path.name,
        "--outfile",
        outfile.name,
        "--sigma",
        sigma,
        "--temp",
        str(float(temperature_k)),
        "--cv",
        cv_names,
        "--bias",
        bias_name,
        "--bin",
        grid_bin,
        "--skiprows",
        str(int(skiprows)),
        "--blocks",
        str(int(blocks)),
    ]
    if grid_min is not None:
        cmd.extend(["--min", grid_min])
    if grid_max is not None:
        cmd.extend(["--max", grid_max])

    subprocess.run(cmd, cwd=str(cwd), check=True)
