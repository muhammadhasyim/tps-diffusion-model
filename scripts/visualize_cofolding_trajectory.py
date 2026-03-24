#!/usr/bin/env python3
"""Visualize diffusion-time trajectories from ``run_cofolding_tps_demo.py`` outputs.

Writes:

1. **Multi-model PDB** — open in PyMOL, VMD, ChimeraX: load as trajectory and scrub frames
   (``Movie > Show state`` / slider).

2. **Last-frame PDB with Boltz topology** (optional) — residue names, atom names, chains,
   and elements from the same ``processed/structures/*.npz`` Boltz used for the run. The
   trajectory may include padded atoms at the end; only the first ``N`` atoms matching the
   structure file are written. Auto-discovers ``boltz_results_*/processed/structures/*.npz``
   next to ``--npz`` when there is exactly one match.

3. Optional **Matplotlib 3D animation** (GIF) — quick sanity check; subsample atoms for speed.

4. Optional **RMSD vs frame / σ** plot from ``trajectory_summary.json`` (curves only, not a
   structure image).

5. Optional **cartoon / ribbon PNG** via **PyMOL** (``--render-cartoon``): ray-traced cartoon
   with ``dss`` secondary structure: helices red, sheets yellow, loops pale green. Requires
   the **PyMOL Python package** in this environment (``pip install pymol-open-source`` or
   ``conda install -c conda-forge pymol-open-source``), or a working ``pymol`` on ``PATH``
   whose launcher can import ``pymol``. Uses ``last_frame.pdb`` by default (run after topology
   export or pass ``--cartoon-pdb``).

Example::

    python scripts/visualize_cofolding_trajectory.py \\
      --npz ./cofolding_tps_out/coords_trajectory.npz \\
      --summary ./cofolding_tps_out/trajectory_summary.json \\
      --pdb-out ./cofolding_tps_out/diffusion_traj.pdb \\
      --last-frame-out ./cofolding_tps_out/last_frame.pdb \\
      --render-cartoon

    # Optional GIF (uses random subsample of atoms for speed)
    python scripts/visualize_cofolding_trajectory.py --npz ... --gif-out traj.gif --gif-atoms 400
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np


def discover_structures_npz(coords_npz: Path, explicit: Path | None) -> Path | None:
    """Return Boltz ``processed/structures/*.npz`` for topology, or ``None`` if ambiguous."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    parent = coords_npz.parent
    candidates = sorted(parent.glob("boltz_results_*/processed/structures/*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def write_last_frame_boltz2_pdb(
    structures_npz: Path,
    frame_coords: np.ndarray,
    out_path: Path,
) -> None:
    """Write one PDB using Boltz-2 :func:`to_pdb` topology and ``frame_coords`` (N, 3) in Å."""
    from boltz.data.types import Coords, Interface, StructureV2
    from boltz.data.write.pdb import to_pdb

    structure = StructureV2.load(structures_npz).remove_invalid_chains()
    n_struct = int(structure.atoms.shape[0])
    fc = np.asarray(frame_coords, dtype=np.float32)
    if fc.shape[0] < n_struct:
        raise ValueError(
            f"Frame has {fc.shape[0]} atoms but Boltz structure has {n_struct}; "
            "cannot align topology to coordinates."
        )
    coord_unpad = fc[:n_struct]
    atoms = structure.atoms.copy()
    atoms["coords"] = coord_unpad
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_arr = np.array([(x,) for x in coord_unpad], dtype=Coords)
    interfaces = np.array([], dtype=Interface)
    new_structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_arr,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_pdb(new_structure, plddts=None, boltz2=True))


def _render_cartoon_pymol_api(
    pdb_path: Path,
    png_path: Path,
    *,
    width: int,
    height: int,
    dpi: int,
    use_ray: bool = True,
) -> None:
    """Headless PyMOL via the ``pymol`` Python package (same interpreter as this script).

    **Warning:** Running embedded PyMOL in the same process as PyTorch/CUDA can segfault
    (OpenGL + driver interaction). Prefer :func:`_render_cartoon_pymol_subprocess` for
    long runs or when ``pymol`` is available on ``PATH``.
    """
    import pymol  # type: ignore[import-untyped]
    from pymol import cmd  # type: ignore[import-untyped]

    pymol.finish_launching(["pymol", "-cqx"])
    try:
        cmd.load(str(pdb_path), "m")
        cmd.hide("everything")
        cmd.show("cartoon", "m")
        cmd.set("cartoon_fancy_helices", 1)
        cmd.dss()
        cmd.color("red", "ss H")
        cmd.color("yellow", "ss S")
        cmd.color("palegreen", "ss L+''")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.set("antialias", 2)
        cmd.orient("m")
        cmd.zoom("m", complete=1)
        cmd.png(str(png_path), width=width, height=height, dpi=dpi, ray=1 if use_ray else 0)
    finally:
        cmd.quit()


def _pymol_subprocess_env(*, prefer_software_gl: bool) -> dict[str, str]:
    """Environment for headless PyMOL; software Mesa stack avoids many NVIDIA EGL segfaults."""
    env = os.environ.copy()
    if prefer_software_gl and "LIBGL_ALWAYS_SOFTWARE" not in env:
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    if env.get("LIBGL_ALWAYS_SOFTWARE") == "1" and "GALLIUM_DRIVER" not in env:
        env["GALLIUM_DRIVER"] = "llvmpipe"
    return env


def _proc_looks_like_segfault(returncode: int | None) -> bool:
    """POSIX subprocess uses -SIG for signal death; shells often report 128 + SIG."""
    if returncode is None:
        return False
    if returncode < 0 and returncode == -signal.SIGSEGV:
        return True
    # 139 == 128 + 11 (seen from some PyMOL wrappers / libc exit codes)
    if returncode in (128 + signal.SIGSEGV, 139):
        return True
    return False


def _render_cartoon_pymol_subprocess(
    pdb_path: Path,
    png_path: Path,
    *,
    width: int,
    height: int,
    dpi: int,
    use_ray: bool = True,
    prefer_software_gl: bool = True,
) -> None:
    """Headless PyMOL in a separate process (``pymol -cqx script.pml``).

    Isolates PyMOL/OpenGL from the CUDA process and avoids intermittent segfaults during
    long TPS jobs.
    """
    import json  # noqa: PLC0415 — only needed for paths containing spaces

    pymol_bin = shutil.which("pymol")
    if not pymol_bin:
        raise RuntimeError(
            "PyMOL executable not on PATH (needed for subprocess cartoon rendering). "
            "Install e.g.: pip install pymol-open-source  OR  conda install -c conda-forge pymol-open-source"
        )

    def _pml_word(path: Path) -> str:
        s = str(path)
        if any(c in s for c in " \t\n\"'\\"):
            return json.dumps(s)
        return s

    pdb_w = _pml_word(pdb_path)
    png_w = _pml_word(png_path)
    ray_flag = 1 if use_ray else 0
    pml = f"""load {pdb_w}
hide everything
show cartoon
set cartoon_fancy_helices, 1
dss
color red, ss H
color yellow, ss S
color palegreen, ss L+''
bg_color white
set ray_opaque_background, 1
set antialias, 2
orient
zoom complete=1
png {png_w}, width={width}, height={height}, dpi={dpi}, ray={ray_flag}
quit
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as tf:
        tf.write(pml)
        pml_path = tf.name
    try:
        env = _pymol_subprocess_env(prefer_software_gl=prefer_software_gl)
        base_cmd = [pymol_bin, "-cqx", pml_path]
        xvfb = shutil.which("xvfb-run")

        def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        combined = ""
        used_xvfb = False
        if xvfb and not os.environ.get("DISPLAY"):
            proc = _run(["xvfb-run", "-a", *base_cmd])
            used_xvfb = True
            combined = (proc.stderr or "") + (proc.stdout or "")
        else:
            proc = _run(base_cmd)
            combined = (proc.stderr or "") + (proc.stdout or "")
        # Real display + GPU drivers often segfault; retry under xvfb + llvmpipe.
        if _proc_looks_like_segfault(proc.returncode) and xvfb and not used_xvfb:
            proc = _run(["xvfb-run", "-a", *base_cmd])
            used_xvfb = True
            combined = (proc.stderr or "") + (proc.stdout or "") + "\n[xvfb retry]\n" + combined

        if proc.returncode != 0:
            hint = ""
            if _proc_looks_like_segfault(proc.returncode):
                hint = (
                    " (SIGSEGV in PyMOL/OpenGL; retried with xvfb-run when possible. "
                    "Try: LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a pymol -cqx …, or conda-forge pymol.)"
                )
            elif "No module named 'pymol'" in combined or "ModuleNotFoundError" in combined:
                hint = (
                    " The pymol launcher on PATH runs a different Python that lacks the "
                    "'pymol' package; install pymol-open-source in the same env as this script."
                )
            raise RuntimeError(f"PyMOL exited {proc.returncode}: {combined[:2000]}{hint}")
        if "ScenePNG-Error" in combined and not png_path.is_file():
            raise RuntimeError(f"PyMOL failed to write PNG: {combined[-1500:]}")
    finally:
        Path(pml_path).unlink(missing_ok=True)
    if not png_path.is_file():
        raise RuntimeError(f"PyMOL did not write {png_path}")


def render_cartoon_png_pymol(
    pdb_path: Path,
    png_path: Path,
    *,
    width: int = 1600,
    height: int = 1600,
    dpi: int = 150,
    force_subprocess: bool = False,
    use_ray: bool = True,
    prefer_software_gl: bool | None = None,
) -> None:
    """Cartoon PNG via PyMOL: ribbon colored by DSSP-like secondary structure.

    Parameters
    ----------
    force_subprocess
        If True, always run ``pymol -cqx`` in a child process (recommended next to a
        long-running PyTorch/CUDA job: in-process PyMOL can segfault). Requires ``pymol``
        on ``PATH``.
    use_ray
        If True, use ray-traced ``png`` (slower; often segfaults in headless/driver setups).
        If False, OpenGL framebuffer capture (less pretty but far more stable for batch jobs).
    prefer_software_gl
        If True, set ``LIBGL_ALWAYS_SOFTWARE=1`` for the PyMOL subprocess (unless already set).
        Default: same as ``force_subprocess`` (Mesa llvmpipe avoids many headless NVIDIA crashes).
    """
    pdb_path = pdb_path.resolve()
    png_path = png_path.resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Cartoon input PDB not found: {pdb_path}")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_software_gl is None:
        prefer_software_gl = bool(force_subprocess)

    if force_subprocess:
        _render_cartoon_pymol_subprocess(
            pdb_path,
            png_path,
            width=width,
            height=height,
            dpi=dpi,
            use_ray=use_ray,
            prefer_software_gl=prefer_software_gl,
        )
        return

    if importlib.util.find_spec("pymol") is not None:
        _render_cartoon_pymol_api(
            pdb_path, png_path, width=width, height=height, dpi=dpi, use_ray=use_ray
        )
        if not png_path.is_file():
            raise RuntimeError(f"PyMOL did not write {png_path}")
        return

    _render_cartoon_pymol_subprocess(
        pdb_path,
        png_path,
        width=width,
        height=height,
        dpi=dpi,
        use_ray=use_ray,
        prefer_software_gl=prefer_software_gl,
    )


def _pdb_line(serial: int, x: float, y: float, z: float, resseq: int, chain: str = "A") -> str:
    """Single ATOM line (PDB fixed columns, element C)."""
    return (
        f"ATOM  {serial:5d}  C   UNK {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def write_multimodel_pdb(
    coords: np.ndarray,
    out_path: Path,
    *,
    chain: str = "A",
) -> None:
    """Write ``coords`` shaped ``(n_frames, n_atoms, 3)`` as PDB MODEL/ENDMDL."""
    n_frames, n_atoms, _ = coords.shape
    lines: list[str] = []
    for f in range(n_frames):
        lines.append(f"MODEL     {f + 1:5d}\n")
        lines.append(f"REMARK  Frame {f} of {n_frames}\n")
        frame = coords[f]
        for a in range(n_atoms):
            x, y, z = float(frame[a, 0]), float(frame[a, 1]), float(frame[a, 2])
            resseq = min(a + 1, 9999)
            lines.append(_pdb_line(a + 1, x, y, z, resseq, chain=chain))
        lines.append("ENDMDL\n")
    lines.append("END\n")
    out_path.write_text("".join(lines))


def load_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    if "coords" not in data:
        raise KeyError(f"{path} must contain 'coords' array")
    arr = np.asarray(data["coords"], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"coords must be (n_frames, n_atoms, 3), got {arr.shape}")
    return arr


def maybe_coords_fallback(
    primary_npz: Path,
    coords: np.ndarray,
    *,
    explicit_fallback: Path | None,
) -> np.ndarray:
    """If ``coords`` are entirely non-finite, load ``coords_trajectory.npz`` (or explicit path)."""
    if np.any(np.isfinite(coords)):
        return coords
    fb = explicit_fallback
    if fb is None:
        fb = primary_npz.parent.parent / "coords_trajectory.npz"
    if not fb.is_file():
        return coords
    try:
        alt = load_npz(fb)
    except Exception as e:
        print(f"Could not load fallback coords {fb}: {e}", file=sys.stderr)
        return coords
    if not np.any(np.isfinite(alt)):
        return coords
    print(
        f"Warning: {primary_npz.name} has no finite coords; using frames from {fb.name}.",
        file=sys.stderr,
    )
    return alt


def plot_summary_plots(summary_path: Path, plot_out: Path) -> None:
    """σ and RMS vs frame from trajectory_summary.json."""
    import matplotlib.pyplot as plt

    with summary_path.open() as f:
        meta = json.load(f)
    frames = meta["frames"]
    sigmas = [float(fr["sigma"]) for fr in frames if fr["sigma"] is not None]
    rms = [fr["rms_vs_frame0_ang"] for fr in frames if fr.get("rms_vs_frame0_ang") is not None]
    idx = list(range(len(frames)))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(idx[: len(sigmas)], sigmas, "b-")
    axes[0].set_ylabel("σ (noise scale)")
    axes[0].set_title("Diffusion schedule (trajectory frames)")
    if rms:
        axes[1].plot(idx[: len(rms)], rms, "g-")
        axes[1].set_ylabel("RMS vs frame 0 (Å)")
    axes[1].set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)


def write_gif_animation(
    coords: np.ndarray,
    out_gif: Path,
    *,
    n_atoms: int = 400,
    seed: int = 0,
    interval_ms: int = 120,
) -> None:
    """3D scatter animation; subsample atoms for speed."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(seed)
    n_total = coords.shape[1]
    if n_atoms < n_total:
        idx = rng.choice(n_total, size=n_atoms, replace=False)
        c = coords[:, idx, :]
    else:
        c = coords

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    lim = float(np.abs(c).max()) * 1.1 + 1e-6

    (scat,) = ax.plot([], [], [], "b.", markersize=2, alpha=0.6)

    def init():
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        return (scat,)

    def update(frame: int):
        x, y, z = c[frame, :, 0], c[frame, :, 1], c[frame, :, 2]
        scat.set_data_3d(x, y, z)
        ax.set_title(f"Frame {frame}/{coords.shape[0]-1}")
        return (scat,)

    anim = FuncAnimation(
        fig,
        update,
        frames=coords.shape[0],
        init_func=init,
        interval=interval_ms,
        blit=False,
    )
    anim.save(out_gif, writer="pillow", fps=max(1, 1000 // interval_ms))
    plt.close(fig)


def render_last_frame_matplotlib_png(
    coords_last: np.ndarray,
    out_png: Path,
    *,
    title: str = "",
) -> None:
    """3D scatter preview of the last frame (Å) when PyMOL is unavailable or crashes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.asarray(coords_last, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords_last must be (n_atoms, 3), got {c.shape}")
    finite = np.all(np.isfinite(c), axis=1)
    c = c[finite]
    if c.shape[0] == 0:
        raise ValueError("No finite (x,y,z) rows in coords_last; cannot render preview.")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    n = c.shape[0]
    sc = ax.scatter(
        c[:, 0],
        c[:, 1],
        c[:, 2],
        c=np.arange(n),
        cmap="viridis",
        s=2,
        alpha=0.65,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.55, label="atom index")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    if title:
        ax.set_title(title)
    max_range = float(
        max(
            c[:, 0].max() - c[:, 0].min(),
            c[:, 1].max() - c[:, 1].min(),
            c[:, 2].max() - c[:, 2].min(),
        )
        / 2.0
        or 1.0
    )
    mid = c.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize coords_trajectory.npz from cofolding TPS demo")
    p.add_argument("--npz", type=Path, required=True, help="coords_trajectory.npz")
    p.add_argument("--pdb-out", type=Path, default=None, help="Write multi-model PDB path")
    p.add_argument(
        "--no-multimodel-pdb",
        action="store_true",
        help="Skip writing the multi-model trajectory PDB (useful for fast batch PNG previews).",
    )
    p.add_argument("--summary", type=Path, default=None, help="trajectory_summary.json for plots")
    p.add_argument("--plot-out", type=Path, default=None, help="Save σ / RMSD plot PNG")
    p.add_argument("--gif-out", type=Path, default=None, help="Save 3D GIF (requires pillow)")
    p.add_argument("--gif-atoms", type=int, default=400, help="Random subsample size for GIF")
    p.add_argument(
        "--structures-npz",
        type=Path,
        default=None,
        help="Boltz processed structures NPZ (same record as the run); default: auto-detect",
    )
    p.add_argument(
        "--last-frame-out",
        type=Path,
        default=None,
        help="Write last-frame PDB with Boltz residue/atom topology (default: last_frame.pdb beside --npz)",
    )
    p.add_argument(
        "--no-last-frame-topology",
        action="store_true",
        help="Skip last-frame Boltz topology PDB even if a structures npz is found",
    )
    p.add_argument(
        "--render-cartoon",
        action="store_true",
        help="Write a cartoon ribbon PNG (PyMOL: helix/sheet/loop colors); needs pymol on PATH",
    )
    p.add_argument(
        "--cartoon-embedded-pymol",
        action="store_true",
        help="Use in-process PyMOL (default: subprocess; embedded often segfaults).",
    )
    p.add_argument(
        "--cartoon-pdb",
        type=Path,
        default=None,
        help="Input PDB for --render-cartoon (default: last_frame.pdb beside --npz)",
    )
    p.add_argument(
        "--cartoon-png",
        type=Path,
        default=None,
        help="Output PNG for --render-cartoon (default: last_frame_cartoon.png beside --npz)",
    )
    p.add_argument("--cartoon-width", type=int, default=1600, help="Cartoon PNG width (pixels)")
    p.add_argument("--cartoon-height", type=int, default=1600, help="Cartoon PNG height (pixels)")
    p.add_argument("--cartoon-dpi", type=int, default=150, help="Cartoon PNG DPI for PyMOL png command")
    p.add_argument(
        "--render-preview-png",
        action="store_true",
        help="Also write a matplotlib 3D scatter of the last frame (no PyMOL).",
    )
    p.add_argument(
        "--preview-png-only",
        action="store_true",
        help="Skip PyMOL; write only the matplotlib preview to --cartoon-png or default path.",
    )
    p.add_argument(
        "--coords-fallback-npz",
        type=Path,
        default=None,
        help=(
            "If coords in --npz are all non-finite, load this NPZ instead (e.g. "
            "coords_trajectory.npz beside the run). Default: try ../coords_trajectory.npz."
        ),
    )
    args = p.parse_args()

    coords = maybe_coords_fallback(args.npz, load_npz(args.npz), explicit_fallback=args.coords_fallback_npz)
    print(f"Loaded coords {coords.shape} (frames, atoms, 3)")

    pdb_out = args.pdb_out
    if pdb_out is None:
        pdb_out = args.npz.with_name(args.npz.stem + ".pdb")
    if not args.no_multimodel_pdb:
        write_multimodel_pdb(coords, pdb_out)
        print(f"Wrote multi-model PDB: {pdb_out.resolve()}")
        print("  Open in PyMOL: File > Open, then display states (bottom right) or: cmd.set('state')")

    if not args.no_last_frame_topology:
        topo = discover_structures_npz(args.npz, args.structures_npz)
        last_out = args.last_frame_out
        if last_out is None and topo is not None:
            last_out = args.npz.with_name("last_frame.pdb")
        if topo is not None and last_out is not None:
            try:
                write_last_frame_boltz2_pdb(topo, coords[-1], last_out)
                print(f"Wrote last-frame topology PDB: {last_out.resolve()}")
                print(f"  (Boltz topology from {topo})")
            except ImportError as e:
                print(f"Skipping last-frame topology PDB (Boltz not importable): {e}", file=sys.stderr)
            except Exception as e:
                print(f"Skipping last-frame topology PDB: {e}", file=sys.stderr)
        elif last_out is not None and topo is None:
            print(
                "Could not find Boltz processed/structures/*.npz (pass --structures-npz or "
                "place a single boltz_results_*/processed/structures/*.npz next to --npz).",
                file=sys.stderr,
            )

    if args.summary and args.summary.is_file():
        plot_path = args.plot_out or args.summary.with_name("trajectory_diagnostics.png")
        plot_summary_plots(args.summary, plot_path)
        print(f"Wrote diagnostics plot: {plot_path.resolve()}")

    c_png_default = args.npz.with_name("last_frame_cartoon.png")
    c_png = args.cartoon_png or c_png_default
    c_pdb = args.cartoon_pdb or args.npz.with_name("last_frame.pdb")

    if args.preview_png_only:
        render_last_frame_matplotlib_png(
            coords[-1],
            c_png,
            title=f"{args.npz.stem} last frame (preview)",
        )
        print(f"Wrote matplotlib preview PNG: {c_png.resolve()}")
    elif args.render_cartoon or args.render_preview_png:
        pymol_ok = False
        if args.render_cartoon:
            try:
                render_cartoon_png_pymol(
                    c_pdb,
                    c_png,
                    width=args.cartoon_width,
                    height=args.cartoon_height,
                    dpi=args.cartoon_dpi,
                    force_subprocess=not args.cartoon_embedded_pymol,
                    use_ray=False,
                )
                print(f"Wrote cartoon ribbon PNG: {c_png.resolve()}")
                pymol_ok = True
            except Exception as e:
                print(f"Cartoon PNG failed: {e}", file=sys.stderr)
                print(
                    "Falling back to matplotlib 3D preview (not a ribbon cartoon).",
                    file=sys.stderr,
                )
                try:
                    render_last_frame_matplotlib_png(
                        coords[-1],
                        c_png,
                        title=f"{args.npz.stem} last frame (preview)",
                    )
                    print(f"Wrote matplotlib preview PNG (PyMOL fallback): {c_png.resolve()}")
                except Exception as e2:
                    print(f"Matplotlib preview failed: {e2}", file=sys.stderr)

        if args.render_preview_png and not args.render_cartoon:
            try:
                render_last_frame_matplotlib_png(
                    coords[-1],
                    c_png,
                    title=f"{args.npz.stem} last frame (preview)",
                )
                print(f"Wrote matplotlib preview PNG: {c_png.resolve()}")
            except Exception as e2:
                print(f"Matplotlib preview failed: {e2}", file=sys.stderr)
        elif args.render_preview_png and args.render_cartoon and pymol_ok:
            prev_path = c_png.with_name(c_png.stem + "_preview.png")
            try:
                render_last_frame_matplotlib_png(
                    coords[-1],
                    prev_path,
                    title=f"{args.npz.stem} last frame (preview)",
                )
                print(f"Wrote matplotlib preview PNG: {prev_path.resolve()}")
            except Exception as e2:
                print(f"Matplotlib preview failed: {e2}", file=sys.stderr)

    if args.gif_out:
        try:
            write_gif_animation(coords, args.gif_out, n_atoms=args.gif_atoms)
            print(f"Wrote GIF: {args.gif_out.resolve()}")
        except Exception as e:
            print(f"GIF failed ({e}). Install pillow: pip install pillow", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
