"""Benchmark runner: deterministic ordering, async concurrency, retries,
timeouts, checkpoint/resume, and output writing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from . import __version__
from .checkpoint import CheckpointStore, RunIdentity
from .config import BenchmarkConfig, ModelConfig
from .providers import BaseProvider, ProviderSettings, create_provider
from .providers.base import ProviderCallError, classify_exception
from .reporting import (
    build_results_document, generate_markdown_report, split_summary_files,
)
from .samples import load_benchmark_file, samples_hash
from .schema import ProviderResult, RunRecord, Sample
from .scoring import classify_failure, score_tool_calls
from .utils.json_utils import atomic_write_json
from .utils.logging import console, get_logger
from .utils.time import local_now_iso, utc_now_iso

log = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/default_system_prompt.txt")


@dataclass(frozen=True)
class RunTask:
    model_cfg: ModelConfig
    sample: Sample
    run_index: int


def load_default_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _record_from_result(sample: Sample, result: ProviderResult,
                        *, save_raw: bool) -> RunRecord:
    scoring = score_tool_calls(
        sample.expected_tool_calls, result.tool_calls, sample.scoring,
        error=result.error, timed_out=result.timed_out,
    )
    failure = classify_failure(scoring, error=result.error,
                               timed_out=result.timed_out)
    return RunRecord(
        model=result.model,
        provider=result.provider,
        sample_id=sample.id,
        run_index=result.run_index,
        success=scoring.success,
        expected_tool_calls=sample.expected_tool_calls,
        actual_tool_calls=result.tool_calls,
        raw_model_output=result.raw_response if save_raw else None,
        text_output=result.text,
        scoring_result=scoring,
        latency_ms=result.latency_ms,
        error=result.error,
        timed_out=result.timed_out,
        failure_type=failure.value if failure else None,
    )


async def _attempt_with_retries(provider: BaseProvider, task: RunTask,
                                system_prompt: str,
                                config: BenchmarkConfig) -> ProviderResult:
    """Call the provider with timeout + retry-on-transient-error. Always
    returns a normalized ProviderResult, never raises."""
    attempts = config.max_retries + 1
    last_error: str | None = None
    timed_out = False
    for attempt in range(attempts):
        try:
            async with asyncio.timeout(config.timeout_seconds):
                return await provider.complete(task.sample, task.run_index,
                                               system_prompt)
        except TimeoutError:
            timed_out = True
            last_error = (f"timed out after {config.timeout_seconds}s "
                          f"(attempt {attempt + 1}/{attempts})")
        except ProviderCallError as exc:
            timed_out = False
            last_error = str(exc)
            if not exc.retryable:
                break
        except Exception as exc:  # SDK-specific exception classes
            timed_out = False
            classified = classify_exception(exc)
            last_error = str(classified)
            if not classified.retryable:
                break
        if attempt < attempts - 1:
            # Deterministic exponential backoff, no jitter.
            await asyncio.sleep(min(2 ** attempt, 30))
    return ProviderResult(
        model=task.model_cfg.name,
        provider=task.model_cfg.provider,
        sample_id=task.sample.id,
        run_index=task.run_index,
        raw_response=None,
        tool_calls=[],
        text="",
        error=last_error,
        timed_out=timed_out,
        latency_ms=0,
    )


def build_task_list(config: BenchmarkConfig,
                    samples: list[Sample]) -> list[RunTask]:
    """Deterministic order: models in config order, samples sorted by id,
    run indices ascending."""
    ordered_samples = sorted(samples, key=lambda s: s.id)
    return [
        RunTask(model_cfg=model_cfg, sample=sample, run_index=run_index)
        for model_cfg in config.models
        for sample in ordered_samples
        for run_index in range(config.runs_per_sample)
    ]


def write_config_snapshot(path: Path, config: BenchmarkConfig, *,
                          config_path: str, config_hash: str,
                          sample_hash: str, started_utc: str,
                          finished_utc: str | None) -> None:
    from .reporting import git_commit_hash
    import platform as _platform

    atomic_write_json(path, {
        "benchmark_name": "tool-call-bench",
        "benchmark_version": __version__,
        "config_path": config_path,
        "config_hash": config_hash,
        "sample_hash": sample_hash,
        "git_commit": git_commit_hash(),
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
        "seed": config.seed,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "config": config.redacted_dump(),
    })


async def run_benchmark(config_path: str, config: BenchmarkConfig, *,
                        resume: bool, force: bool) -> dict[str, Any]:
    started_utc = utc_now_iso()

    # 1. Load and validate samples — abort before any API call on failure.
    samples = load_benchmark_file(config.benchmark)
    sample_hash = samples_hash(samples)
    config_hash = config.config_hash()
    log.info("Loaded %d samples from %s", len(samples), config.benchmark)

    # 2. Construct providers and verify credentials up front.
    settings = ProviderSettings(
        temperature=config.temperature,
        allow_text_tool_call_fallback=config.allow_text_tool_call_fallback,
    )
    providers: dict[str, BaseProvider] = {}
    for model_cfg in config.models:
        provider = create_provider(model_cfg, settings)
        provider.check_credentials()
        providers[model_cfg.key] = provider
    log.info("Credentials verified for %d model(s)", len(config.models))

    # 3. Open the checkpoint store.
    output_path = Path(config.output)
    output_dir = output_path.parent
    checkpoint_dir = output_dir / "checkpoints"
    identity = RunIdentity(
        benchmark_version=__version__,
        config_hash=config_hash,
        sample_hash=sample_hash,
    )
    store = CheckpointStore.open(checkpoint_dir, identity,
                                 resume=resume, force=force)
    if store.completed():
        log.info("Resuming: %d completed run(s) found in checkpoint",
                 len(store.completed()))

    # 4. Snapshot the redacted config before making any calls.
    snapshot_path = output_dir / "run_config.snapshot.json"
    write_config_snapshot(snapshot_path, config, config_path=config_path,
                          config_hash=config_hash, sample_hash=sample_hash,
                          started_utc=started_utc, finished_utc=None)

    # 5. Build the deterministic task list and skip completed runs.
    tasks = build_task_list(config, samples)
    pending = [
        t for t in tasks
        if not store.has(store.key_for(t.model_cfg.provider, t.model_cfg.key,
                                       t.sample.id, t.run_index))
    ]
    log.info("Total runs: %d — already completed: %d — to run: %d",
             len(tasks), len(tasks) - len(pending), len(pending))

    default_system = load_default_system_prompt()
    semaphore = asyncio.Semaphore(config.concurrency)
    checkpoint_lock = asyncio.Lock()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    async def execute(task: RunTask, progress_task_id) -> None:
        provider = providers[task.model_cfg.key]
        system_prompt = task.sample.system_prompt or default_system
        async with semaphore:
            result = await _attempt_with_retries(provider, task,
                                                 system_prompt, config)
        record = _record_from_result(task.sample, result,
                                     save_raw=config.save_raw_responses)
        key = store.key_for(task.model_cfg.provider, task.model_cfg.key,
                            task.sample.id, task.run_index)
        async with checkpoint_lock:
            # Persist after EVERY completed run so interruption is safe.
            store.add(key, record)
        progress.update(progress_task_id, advance=1)

    if pending:
        with progress:
            progress_task_id = progress.add_task(
                "Running benchmark", total=len(pending))
            try:
                await asyncio.gather(
                    *(execute(t, progress_task_id) for t in pending))
            except (KeyboardInterrupt, asyncio.CancelledError):
                log.warning(
                    "Interrupted — progress is checkpointed. Re-run the same "
                    "command to resume."
                )
                raise

    finished_utc = utc_now_iso()

    # 6. Assemble results from the checkpoint (covers resumed runs too).
    records = list(store.completed().values())
    results = build_results_document(
        config=config,
        config_path=config_path,
        config_hash=config_hash,
        samples=samples,
        sample_hash=sample_hash,
        sample_file=config.benchmark,
        records=records,
        started_utc=started_utc,
        finished_utc=finished_utc,
        generated_utc=finished_utc,
    )

    # 7. Write every output file from the single results document.
    atomic_write_json(output_path, results)
    for filename, payload in split_summary_files(results).items():
        atomic_write_json(output_dir / filename, payload)
    report_path = Path(config.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        generate_markdown_report(results, local_date=local_now_iso()),
        encoding="utf-8",
    )
    write_config_snapshot(snapshot_path, config, config_path=config_path,
                          config_hash=config_hash, sample_hash=sample_hash,
                          started_utc=started_utc, finished_utc=finished_utc)

    log.info("Results written to %s", output_path)
    log.info("Report written to %s", report_path)
    return results
