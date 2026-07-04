"""``benchmark`` command-line interface."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import __version__
from .checkpoint import CheckpointConflictError
from .config import BenchmarkConfig
from .reporting import generate_markdown_report
from .samples import (
    EXPECTED_SAMPLE_COUNT, SampleValidationError, generate_sample_files,
    load_benchmark_file, samples_hash,
)
from .utils.env import MissingCredentialError
from .utils.json_utils import load_json_file
from .utils.logging import console
from .utils.time import local_now_iso

app = typer.Typer(
    name="benchmark",
    help="tool-call-bench: measure LLM tool-calling accuracy.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)


@app.callback()
def _main_callback() -> None:
    """tool-call-bench CLI."""


@app.command()
def version() -> None:
    """Print the benchmark version."""
    console.print(__version__)


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False,
                                       help="Path to a benchmark config JSON."),
    resume: Optional[bool] = typer.Option(
        None, "--resume/--no-resume",
        help="Resume from the checkpoint (default from config: resume=true)."),
    force: bool = typer.Option(
        False, "--force",
        help="Discard the checkpoint and rerun everything."),
) -> None:
    """Run the benchmark described by CONFIG_PATH."""
    try:
        config = BenchmarkConfig.load(config_path)
    except Exception as exc:
        _fail(f"invalid config {config_path}: {exc}")
        return
    effective_resume = config.resume if resume is None else resume
    effective_force = config.force or force
    try:
        results = asyncio.run(run_command(str(config_path), config,
                                          resume=effective_resume,
                                          force=effective_force))
    except SampleValidationError as exc:
        _fail(str(exc))
        return
    except MissingCredentialError as exc:
        _fail(str(exc))
        return
    except CheckpointConflictError as exc:
        _fail(str(exc))
        return
    except KeyboardInterrupt:
        _fail("interrupted — rerun the same command to resume", code=130)
        return
    _print_summary_tables(results)


async def run_command(config_path: str, config: BenchmarkConfig, *,
                      resume: bool, force: bool):
    from .runner import run_benchmark  # deferred: heavy SDK imports

    return await run_benchmark(config_path, config, resume=resume,
                               force=force)


@app.command("generate-samples")
def generate_samples(
    out_dir: Path = typer.Option(Path("benchmark_samples"), "--out-dir",
                                 help="Directory for generated samples."),
) -> None:
    """Generate the 67 sample JSON files and full_benchmark.json."""
    combined = generate_sample_files(out_dir)
    samples = load_benchmark_file(combined)
    console.print(
        f"[green]Wrote {len(samples)} samples[/green] to "
        f"{out_dir / 'tool_calling'} and combined file {combined}"
    )
    console.print(f"Sample hash: {samples_hash(samples)}")


@app.command("validate-samples")
def validate_samples_cmd(
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                          help="Path to full_benchmark.json."),
) -> None:
    """Validate every sample; fails before any API call on a malformed one."""
    try:
        samples = load_benchmark_file(benchmark_file)
    except SampleValidationError as exc:
        _fail(str(exc))
        return
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        console.print(
            f"[yellow]warning:[/yellow] expected {EXPECTED_SAMPLE_COUNT} "
            f"samples, found {len(samples)}"
        )
    by_category: dict[str, int] = {}
    for s in samples:
        by_category[s.category.value] = by_category.get(s.category.value, 0) + 1
    console.print(f"[green]OK[/green] — {len(samples)} valid samples")
    for category, count in sorted(by_category.items()):
        console.print(f"  {category}: {count}")
    console.print(f"Sample hash: {samples_hash(samples)}")


@app.command()
def report(
    results_path: Path = typer.Argument(..., exists=True, dir_okay=False,
                                        help="Path to outputs/results.json."),
    out: Path = typer.Option(Path("reports/tool_call_report.md"), "--out",
                             help="Where to write the Markdown report."),
) -> None:
    """Generate the Markdown report from an existing results.json."""
    results = load_json_file(results_path)
    if not isinstance(results, dict) or "summary" not in results:
        _fail(f"{results_path} does not look like a results.json file")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        generate_markdown_report(results, local_date=local_now_iso()),
        encoding="utf-8",
    )
    console.print(f"[green]Report written[/green] to {out}")


@app.command()
def summarize(
    results_path: Path = typer.Argument(..., exists=True, dir_okay=False,
                                        help="Path to outputs/results.json."),
) -> None:
    """Print model performance summaries to the terminal."""
    results = load_json_file(results_path)
    if not isinstance(results, dict) or "model_summaries" not in results:
        _fail(f"{results_path} does not look like a results.json file")
        return
    _print_summary_tables(results)


def _print_summary_tables(results: dict) -> None:
    s = results["summary"]
    console.print()
    console.print(
        f"[bold]tool-call-bench[/bold] v{s['benchmark_version']} — "
        f"{s['sample_count']} samples × {s['runs_per_sample']} runs × "
        f"{s['model_count']} model(s) = {s['total_runs_all_models']} runs"
    )

    ranking = Table(title="Overall Ranking", show_lines=False)
    for column, justify in (("Rank", "right"), ("Model", "left"),
                            ("Provider", "left"), ("Successful", "right"),
                            ("Total", "right"), ("Success Rate", "right")):
        ranking.add_column(column, justify=justify)
    ranked = sorted(results["model_summaries"],
                    key=lambda m: (-m["success_rate"], m["model"],
                                   m["provider"]))
    for i, m in enumerate(ranked, start=1):
        ranking.add_row(str(i), m["model"], m["provider"],
                        str(m["successful_runs"]), str(m["total_runs"]),
                        m["success_rate_percent"])
    console.print(ranking)

    categories = Table(title="Per-Category Success Rates")
    for column, justify in (("Model", "left"), ("Provider", "left"),
                            ("Category", "left"), ("Success Rate", "right")):
        categories.add_column(column, justify=justify)
    for c in results["category_summaries"]:
        categories.add_row(c["model"], c["provider"], c["category"],
                           c["success_rate_percent"])
    console.print(categories)

    failure_counts = results["failure_statistics"]["failure_type_counts"]
    failures = Table(title="Failure Types (all models)")
    failures.add_column("Failure Type", justify="left")
    failures.add_column("Count", justify="right")
    for failure_type, count in failure_counts.items():
        failures.add_row(failure_type, str(count))
    console.print(failures)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
