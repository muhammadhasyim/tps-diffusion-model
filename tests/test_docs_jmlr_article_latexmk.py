"""Regression: JMLR article TeX must compile (latexmk).

Skipped when ``latexmk`` is not installed (e.g. minimal CI images).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX = REPO_ROOT / "docs" / "paper_tps_generative_models_SI.tex"


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk not installed")
def test_jmlr_article_latexmk_builds() -> None:
    """``latexmk -cd -pdf`` succeeds for the main JMLR article driver."""
    assert TEX.is_file(), f"missing {TEX}"
    proc = subprocess.run(
        [
            "latexmk",
            "-cd",
            "-pdf",
            "-interaction=batchmode",
            str(TEX.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
    assert proc.returncode == 0, "latexmk failed for docs/paper_tps_generative_models_SI.tex"
    pdf = REPO_ROOT / "docs" / "paper_tps_generative_models_SI.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 10_000, f"missing or tiny PDF: {pdf}"
