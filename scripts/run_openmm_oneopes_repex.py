#!/usr/bin/env python3
"""OneOPES Hamiltonian replica exchange driver (OpenMM + PLUMED)."""

from __future__ import annotations

import argparse
import sys

from genai_tps.simulation.openmm_md_runner import build_opes_md_argument_parser
from genai_tps.simulation.oneopes_repex import (
    register_oneopes_repex_arguments,
    validate_oneopes_repex_cli_args,
)
from genai_tps.simulation.repex_distributed import run_distributed_repex


def _build_parser() -> argparse.ArgumentParser:
    parser = build_opes_md_argument_parser()
    register_oneopes_repex_arguments(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    validate_oneopes_repex_cli_args(args, parser)
    run_distributed_repex(args)


if __name__ == "__main__":
    main()
