#!/usr/bin/env python3
"""CLI for OPES-biased OpenMM MD — parses ``argv`` and runs :func:`run_opes_md`."""

from __future__ import annotations

import sys

from genai_tps.simulation.openmm_md_runner import parse_opes_md_args, run_opes_md


def main() -> None:
    run_opes_md(parse_opes_md_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
