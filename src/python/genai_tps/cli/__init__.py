"""Typer CLI for genai-tps (sample → train → eval pipeline)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _scripts_dir() -> Path:
    return _REPO_ROOT / "scripts"


def _run_script(script_name: str, extra: list[str]) -> None:
    target = _scripts_dir() / script_name
    if not target.is_file():
        raise typer.BadParameter(f"Missing script: {target}")
    cmd = [sys.executable, str(target), *extra]
    raise typer.Exit(subprocess.call(cmd, env=os.environ.copy()))


sample_app = typer.Typer(help="Path sampling (OPES-TPS, dataset assembly)")
train_app = typer.Typer(help="Weighted DSM training")
eval_app = typer.Typer(help="Evaluation & dashboards")
pipeline_app = typer.Typer(help="End-to-end orchestration (calls scripts)")


@sample_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def sample_run(ctx: typer.Context) -> None:
    """Delegate to ``scripts/run_opes_tps.py`` (pass-through argv)."""
    _run_script("run_opes_tps.py", list(ctx.args))


@sample_app.command(
    "assemble",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def sample_assemble(ctx: typer.Context) -> None:
    """Assemble per-step WDSM shards (``assemble_wdsm_dataset.py``)."""
    _run_script("assemble_wdsm_dataset.py", list(ctx.args))


@train_app.command(
    "fit",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_fit(ctx: typer.Context) -> None:
    """Weighted DSM fine-tuning (``train_weighted_dsm.py``)."""
    _run_script("train_weighted_dsm.py", list(ctx.args))


@train_app.command(
    "split",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_split(ctx: typer.Context) -> None:
    """Train/val split for NPZ datasets."""
    _run_script("split_wdsm_dataset.py", list(ctx.args))


@eval_app.command(
    "generate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_generate(ctx: typer.Context) -> None:
    """Generate structures / metrics (``evaluate_wdsm_model.py``)."""
    _run_script("evaluate_wdsm_model.py", list(ctx.args))


@eval_app.command(
    "compare",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_compare(ctx: typer.Context) -> None:
    """Multi-model comparison dashboard."""
    _run_script("compare_models.py", list(ctx.args))


@pipeline_app.callback(invoke_without_command=True)
def pipeline_main(ctx: typer.Context) -> None:
    """Run stage scripts sequentially with user-provided commands (advanced)."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("Use genai-tps sample|train|eval subcommands, or wire scripts in examples/.", err=True)
    raise typer.Exit(code=2)


app = typer.Typer(help="Generative TPS / Boltz path-sampling toolkit")
app.add_typer(sample_app, name="sample")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(pipeline_app, name="pipeline")


def main() -> None:
    """Console entrypoint for ``genai-tps``."""
    app()


if __name__ == "__main__":
    main()
