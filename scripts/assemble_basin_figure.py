"""
assemble_basin_figure.py
Assembles a 4-column x 3-row composite figure of basin-representative structures.

Each column corresponds to one of the four TPS-OPES basins (B0–B3), and each row
holds one of the three representative structures rendered by render_tps_checkpoint_cartoons.py.
Column headers display the basin label with approximate (s1, s2) center coordinates.

Usage:
    python scripts/assemble_basin_figure.py \
        --png-root opes_tps_out_case1_2d/pymol_basins \
        --out docs/figs/basin_structural_ensemble.png
"""

import argparse
import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


BASIN_DEFS = [
    {
        "name": "B0_native",
        "label": r"B$_0$ (Native)" + "\n" + r"$s_1\approx0.07\,\AA$, $s_2\approx5.37\,\AA$",
        "subdir": "B0_native",
    },
    {
        "name": "B1_intermediate",
        "label": r"B$_1$ (Intermediate)" + "\n" + r"$s_1\approx0.35\,\AA$, $s_2\approx5.47\,\AA$",
        "subdir": "B1_intermediate",
    },
    {
        "name": "B2_displaced",
        "label": r"B$_2$ (Displaced)" + "\n" + r"$s_1\approx0.75\,\AA$, $s_2\approx5.56\,\AA$",
        "subdir": "B2_displaced",
    },
    {
        "name": "B3_far_tail",
        "label": r"B$_3$ (Far tail)" + "\n" + r"$s_1\approx1.10\,\AA$, $s_2\approx5.48\,\AA$",
        "subdir": "B3_far_tail",
    },
]


def load_pngs(png_dir: pathlib.Path, n: int = 3) -> list:
    """Load up to *n* PNG images sorted by filename from *png_dir*."""
    pngs = sorted(png_dir.glob("*.png"))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG files found in {png_dir}")
    return [mpimg.imread(str(p)) for p in pngs[:n]]


def assemble_figure(
    png_root: pathlib.Path,
    out_path: pathlib.Path,
    n_rows: int = 3,
    dpi: int = 300,
    row_labels: Optional[list[str]] = None,
) -> None:
    """Create and save the composite publication figure.

    Parameters
    ----------
    png_root:
        Directory containing per-basin subdirectories (each with a ``png/`` subdirectory).
    out_path:
        Destination path for the assembled PNG.
    n_rows:
        Number of structure rows per basin column (default 3).
    dpi:
        Output resolution in dots per inch.
    row_labels:
        Optional list of left-side row labels (length must equal n_rows).
    """
    n_cols = len(BASIN_DEFS)

    # Load images
    images: list[list] = []
    for bd in BASIN_DEFS:
        png_dir = png_root / bd["subdir"] / "png"
        images.append(load_pngs(png_dir, n_rows))

    # Determine cell size (use first image as reference)
    ref_img = images[0][0]
    img_h, img_w = ref_img.shape[:2]
    cell_w = img_w / dpi  # inches
    cell_h = img_h / dpi

    # Layout constants (inches)
    header_h = 0.55          # column header height
    row_label_w = 0.0        # no row labels
    fig_w = n_cols * cell_w + row_label_w + 0.10
    fig_h = n_rows * cell_h + header_h + 0.10

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")

    # Grid: header_h_frac + n_rows cells
    header_frac = header_h / fig_h
    body_frac = 1.0 - header_frac - 0.02

    gs = fig.add_gridspec(
        n_rows + 1,
        n_cols,
        height_ratios=[header_h] + [cell_h] * n_rows,
        hspace=0.02,
        wspace=0.02,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
    )

    # Column headers
    for col_idx, bd in enumerate(BASIN_DEFS):
        ax_hdr = fig.add_subplot(gs[0, col_idx])
        ax_hdr.axis("off")
        ax_hdr.text(
            0.5, 0.5,
            bd["label"],
            transform=ax_hdr.transAxes,
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            multialignment="center",
        )

    # Structure panels
    for col_idx, bd in enumerate(BASIN_DEFS):
        imgs = images[col_idx]
        for row_idx in range(n_rows):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.imshow(imgs[row_idx])
            ax.axis("off")
            # Row-number label on leftmost column
            if col_idx == 0:
                ax.text(
                    -0.04, 0.5,
                    f"{row_idx + 1}",
                    transform=ax.transAxes,
                    ha="right", va="center",
                    fontsize=6, color="#555555",
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved composite figure -> {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png-root",
        type=pathlib.Path,
        default=pathlib.Path("opes_tps_out_case1_2d/pymol_basins"),
        help="Root directory containing per-basin PNG subdirectories.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("docs/figs/basin_structural_ensemble.png"),
        help="Output path for the composite figure.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default 300).")
    parser.add_argument("--n-rows", type=int, default=3, help="Number of structure rows.")
    args = parser.parse_args()

    assemble_figure(
        png_root=args.png_root,
        out_path=args.out,
        n_rows=args.n_rows,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
