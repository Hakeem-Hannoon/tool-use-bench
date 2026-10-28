"""Results aggregation and report generation.

``build_results_document`` produces the complete ``outputs/results.json``
structure. The Markdown report and the split summary files are derived from
that document — a single calculation path, so the ``.md`` report and the
``.json`` results can never disagree.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from collections import Counter
from typing import Any

from . import BENCHMARK_NAME, __version__
from .config import BenchmarkConfig, ModelConfig
from .schema import FailureType, RunRecord, Sample

_FAILURE_KEYS = [ft.value for ft in FailureType]

_FAILURE_COUNT_FIELDS = {
    FailureType.MISSING_TOOL_CALL.value: "missing_tool_call_failures",
    FailureType.WRONG_TOOL_NAME.value: "wrong_tool_name_failures",
    FailureType.SCHEMA_VIOLATION.value: "schema_violation_failures",
    FailureType.WRONG_ARGUMENT.value: "wrong_argument_failures",
    FailureType.EXTRA_TOOL_CALL.value: "extra_tool_call_failures",
    FailureType.ORDERING_ERROR.value: "ordering_failures",
    FailureType.WORLD_MISMATCH.value: "world_mismatch_failures",
    FailureType.PROTOCOL_ERROR.value: "protocol_error_failures",
    FailureType.PROVIDER_ERROR.value: "provider_error_failures",
    FailureType.TIMEOUT.value: "timeout_failures",
}


def rate_and_percent(successes: int, total: int) -> tuple[float, str]:
    rate = (successes / total) if total else 0.0
    return round(rate, 10), f"{rate * 100:.2f}%"


def percentile(values: list[int], pct: float) -> int | None:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = out.stdout.strip()
        return commit or None
    except (OSError, subprocess.SubprocessError):
        return None


def record_model_key(record: RunRecord) -> str:
    """Group key for a run: its model entry's key (older records fall back
    to provider/model)."""
    return record.model_key or f"{record.provider}/{record.model}"


def _cost_stats(records: list[RunRecord]) -> dict[str, Any]:
    """Token, cost, latency, and step statistics for a set of runs."""
    n = len(records)
    successes = sum(1 for r in records if r.success)
    priced = [r.cost_usd for r in records if r.cost_usd is not None]
    total_cost = round(sum(priced), 8) if priced and len(priced) == n else None
    latencies = [r.latency_ms for r in records if r.error is None and not r.timed_out]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "avg_input_tokens": mean([r.usage.input_tokens for r in records]),
        "avg_cached_input_tokens": mean([r.usage.cached_input_tokens for r in records]),
        "avg_output_tokens": mean([r.usage.output_tokens for r in records]),
        "total_cost_usd": total_cost,
        "cost_per_run_usd": round(total_cost / n, 8) if total_cost is not None and n else None,
        "cost_per_successful_run_usd": (round(total_cost / successes, 8)
                                        if total_cost is not None and successes else None),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "avg_steps": mean([r.step_count for r in records]),
        "recoveries": sum(r.recoveries for r in records),
    }


def _entry_label(key: str, cfg: ModelConfig | None,
                 records: list[RunRecord]) -> dict[str, str]:
    if cfg is not None:
        return {"model": cfg.name, "provider": cfg.provider,
                "model_key": key, "label": cfg.display,
                "protocol": cfg.protocol}
    first = records[0] if records else None
    return {"model": first.model if first else key,
            "provider": first.provider if first else "",
            "model_key": key, "label": (first.label or first.model) if first else key,
            "protocol": first.protocol if first else "native"}


def workload_picks(category_summaries: list[dict[str, Any]],
                   bar: float) -> list[dict[str, Any]]:
    """Per category: the cheapest model that clears ``bar``.

    Among models at or above the bar, the lowest cost per run wins (models
    without pricing rank after priced ones, then by success rate). When no
    model clears the bar, the most reliable one is reported as the best
    available, flagged ``clears_bar: false``.
    """
    picks: list[dict[str, Any]] = []
    for category in sorted({c["category"] for c in category_summaries}):
        rows = [c for c in category_summaries if c["category"] == category]
        eligible = [c for c in rows if c["success_rate"] >= bar]
        if eligible:
            best = min(eligible, key=lambda c: (
                c["cost_per_run_usd"] is None,
                c["cost_per_run_usd"] or 0.0,
                -c["success_rate"], c["label"]))
        else:
            best = max(rows, key=lambda c: (c["success_rate"],
                                            -(c["cost_per_run_usd"] or 0.0)))
        picks.append({
            "category": category,
            "clears_bar": bool(eligible),
            "model_key": best["model_key"],
            "label": best["label"],
            "success_rate": best["success_rate"],
            "success_rate_percent": best["success_rate_percent"],
            "cost_per_run_usd": best["cost_per_run_usd"],
            "eligible_models": len(eligible),
        })
    return picks


def build_results_document(
    *,
    config: BenchmarkConfig,
    config_path: str,
    config_hash: str,
    samples: list[Sample],
    sample_hash: str,
    sample_file: str,
    records: list[RunRecord],
    started_utc: str,
    finished_utc: str,
    generated_utc: str,
) -> dict[str, Any]:
    sample_by_id = {s.id: s for s in samples}
    cfg_by_key = {m.key: m for m in config.models}
    model_order = [m.key for m in config.models]
    for record in records:  # records from entries no longer in the config
        key = record_model_key(record)
        if key not in cfg_by_key and key not in model_order:
            model_order.append(key)

    # Deterministic run order: config model order, then sample id, run index.
    order_index = {key: i for i, key in enumerate(model_order)}
    records = sorted(
        records,
        key=lambda r: (order_index.get(record_model_key(r), 10**9),
                       r.sample_id, r.run_index),
    )
    by_model: dict[str, list[RunRecord]] = {k: [] for k in model_order}
    for record in records:
        by_model[record_model_key(record)].append(record)

    # ------------------------------------------------------------------
    # Per-model aggregation
    # ------------------------------------------------------------------
    model_summaries: list[dict[str, Any]] = []
    labels: dict[str, dict[str, str]] = {}
    for key in model_order:
        model_records = by_model[key]
        label = _entry_label(key, cfg_by_key.get(key), model_records)
        labels[key] = label
        total = len(model_records)
        successes = sum(1 for r in model_records if r.success)
        failure_counts = Counter(
            r.failure_type for r in model_records
            if not r.success and r.failure_type
        )
        per_sample_rates: list[float] = []
        perfect = 0
        with_failure = 0
        for sid in sorted({r.sample_id for r in model_records}):
            sample_runs = [r for r in model_records if r.sample_id == sid]
            wins = sum(1 for r in sample_runs if r.success)
            per_sample_rates.append(wins / len(sample_runs))
            if wins == len(sample_runs):
                perfect += 1
            else:
                with_failure += 1
        rate, percent = rate_and_percent(successes, total)
        avg_rate = (sum(per_sample_rates) / len(per_sample_rates)
                    if per_sample_rates else 0.0)
        summary: dict[str, Any] = {
            **label,
            "successful_runs": successes,
            "failed_runs": total - successes,
            "total_runs": total,
            "success_rate": rate,
            "success_rate_percent": percent,
            "samples_passed_all_30_runs": perfect,
            "samples_with_at_least_one_failure": with_failure,
            "average_success_rate_per_sample": round(avg_rate, 4),
        }
        for failure_value, field in _FAILURE_COUNT_FIELDS.items():
            summary[field] = failure_counts.get(failure_value, 0)
        summary.update(_cost_stats(model_records))
        model_summaries.append(summary)

    # ------------------------------------------------------------------
    # Per-category and per-sample aggregation
    # ------------------------------------------------------------------
    category_summaries: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    for key in model_order:
        model_records = by_model[key]
        label = labels[key]
        categories = sorted({
            sample_by_id[r.sample_id].category
            for r in model_records if r.sample_id in sample_by_id
        })
        for category in categories:
            cat_records = [
                r for r in model_records
                if r.sample_id in sample_by_id
                and sample_by_id[r.sample_id].category == category
            ]
            wins = sum(1 for r in cat_records if r.success)
            rate, percent = rate_and_percent(wins, len(cat_records))
            stats = _cost_stats(cat_records)
            category_summaries.append({
                **label,
                "category": category,
                "successful_runs": wins,
                "failed_runs": len(cat_records) - wins,
                "total_runs": len(cat_records),
                "success_rate": rate,
                "success_rate_percent": percent,
                "cost_per_run_usd": stats["cost_per_run_usd"],
                "latency_p50_ms": stats["latency_p50_ms"],
            })
        for sid in sorted({r.sample_id for r in model_records}):
            sample_runs = [r for r in model_records if r.sample_id == sid]
            wins = sum(1 for r in sample_runs if r.success)
            rate, percent = rate_and_percent(wins, len(sample_runs))
            sample = sample_by_id.get(sid)
            sample_summaries.append({
                **label,
                "sample_id": sid,
                "category": sample.category if sample else "unknown",
                "difficulty": sample.difficulty.value if sample else "unknown",
                "successful_runs": wins,
                "failed_runs": len(sample_runs) - wins,
                "total_runs": len(sample_runs),
                "success_rate": rate,
                "success_rate_percent": percent,
            })

    # ------------------------------------------------------------------
    # Failure statistics across all models
    # ------------------------------------------------------------------
    missing_counter: Counter[str] = Counter()
    wrong_arg_counter: Counter[str] = Counter()
    schema_counter: Counter[str] = Counter()
    failure_type_counts = {key: 0 for key in _FAILURE_KEYS}
    for record in records:
        if record.success:
            continue
        if record.failure_type in failure_type_counts:
            failure_type_counts[record.failure_type] += 1
        missing_counter.update(record.scoring_result.missing_tool_calls)
        wrong_arg_counter.update(
            f"{e.tool_name}.{e.argument_path}"
            for e in record.scoring_result.argument_errors
        )
        schema_counter.update(
            f"{e.tool_name}.{e.argument_path or '<root>'}"
            for e in record.scoring_result.schema_errors
        )

    def _ranked(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{key_name: name, "count": count} for name, count in ranked]

    total_runs_all = len(records)
    runs_per_model = (config.runs_per_sample * len(samples))

    document: dict[str, Any] = {
        "summary": {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_version": __version__,
            "generated_utc": generated_utc,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "config_path": config_path,
            "config_hash": config_hash,
            "sample_file": sample_file,
            "sample_hash": sample_hash,
            "git_commit": git_commit_hash(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": config.seed,
            "runs_per_sample": config.runs_per_sample,
            "sample_count": len(samples),
            "model_count": len(config.models),
            "total_runs_per_model": runs_per_model,
            "total_runs_all_models": total_runs_all,
            "reliability_bar": config.reliability_bar,
        },
        "model_summaries": model_summaries,
        "category_summaries": category_summaries,
        "sample_summaries": sample_summaries,
        "workload_picks": workload_picks(category_summaries,
                                         config.reliability_bar),
        "failure_statistics": {
            "most_common_missing_tools": _ranked(missing_counter, "tool_name"),
            "most_common_wrong_arguments": _ranked(wrong_arg_counter,
                                                   "argument_path"),
            "most_common_schema_violations": _ranked(schema_counter,
                                                     "argument_path"),
            "failure_type_counts": failure_type_counts,
        },
        "model_configs": config.redacted_dump()["models"],
        "runs": [r.model_dump(mode="json") for r in records],
    }
    return document


def split_summary_files(results: dict[str, Any]) -> dict[str, Any]:
    """Derive the auxiliary JSON output files from the results document."""
    failures = [run for run in results["runs"] if not run["success"]]
    return {
        "model_summaries.json": {
            "summary": results["summary"],
            "model_summaries": results["model_summaries"],
        },
        "sample_summaries.json": {
            "summary": results["summary"],
            "sample_summaries": results["sample_summaries"],
        },
        "category_summaries.json": {
            "summary": results["summary"],
            "category_summaries": results["category_summaries"],
        },
        "failures.json": {
            "summary": results["summary"],
            "failure_statistics": results["failure_statistics"],
            "failed_runs": failures,
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.6f}"


def _ms(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _label(entry: dict[str, Any]) -> str:
    return entry.get("label") or entry["model"]


def _table(headers: list[str], aligns: list[str],
           rows: list[list[Any]]) -> str:
    sep = {"l": "---", "r": "---:"}
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(sep[a] for a in aligns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    return "\n".join(lines)


def generate_markdown_report(results: dict[str, Any], *,
                             local_date: str | None = None) -> str:
    s = results["summary"]
    lines: list[str] = []
    add = lines.append

    add("# Tool Call Benchmark Report")
    add("")
    add(f"Generated UTC: {s['generated_utc']}  ")
    if local_date:
        add(f"Generated local: {local_date}  ")
    if s.get("git_commit"):
        add(f"Git commit: {s['git_commit']}  ")
    add(f"Benchmark version: {s['benchmark_version']}  ")
    add(f"Config file: {s['config_path']}  ")
    add(f"Config hash: `{s['config_hash']}`  ")
    add(f"Sample file: {s['sample_file']}  ")
    add(f"Sample hash: `{s['sample_hash']}`  ")
    add(f"Models: {s['model_count']}  ")
    add(f"Samples: {s['sample_count']}  ")
    add(f"Runs per sample: {s['runs_per_sample']}  ")
    add(f"Total runs per model: {s['total_runs_per_model']}  ")
    add(f"Total runs overall: {s['total_runs_all_models']}  ")
    add("")

    add("## Overall Ranking")
    add("")
    ranked = sorted(
        results["model_summaries"],
        key=lambda m: (-m["success_rate"], _label(m), m["provider"]),
    )
    add(_table(
        ["Rank", "Model", "Provider", "Successful Runs", "Total Runs",
         "Success Rate"],
        ["r", "l", "l", "r", "r", "r"],
        [[i + 1, _label(m), m["provider"], m["successful_runs"],
          m["total_runs"], m["success_rate_percent"]]
         for i, m in enumerate(ranked)],
    ))
    add("")

    add("## Cost & Latency")
    add("")
    add("Tokens are per run (all steps of an episode). Cost comes from each "
        "model's configured `pricing`; `—` means no pricing was configured. "
        "Latency covers successful provider calls only. Recoveries count "
        "router nudges (invalid JSON, unknown tool) that cost an extra call.")
    add("")
    add(_table(
        ["Model", "Protocol", "Avg Input Tok", "Avg Cached Tok",
         "Avg Output Tok", "Cost / Run", "Cost / Success", "p50 ms",
         "p95 ms", "Avg Steps", "Recoveries"],
        ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        [[_label(m), m.get("protocol", "native"),
          f"{m.get('avg_input_tokens', 0):,.0f}",
          f"{m.get('avg_cached_input_tokens', 0):,.0f}",
          f"{m.get('avg_output_tokens', 0):,.0f}",
          _usd(m.get("cost_per_run_usd")),
          _usd(m.get("cost_per_successful_run_usd")),
          _ms(m.get("latency_p50_ms")), _ms(m.get("latency_p95_ms")),
          m.get("avg_steps", 0), m.get("recoveries", 0)]
         for m in ranked],
    ))
    add("")

    picks = results.get("workload_picks") or []
    if picks:
        bar = s.get("reliability_bar", 0.95)
        add("## Workload Picks")
        add("")
        add(f"For each category, the cheapest model whose success rate is at "
            f"least {bar * 100:.0f}%. When none clears the bar, the most "
            f"reliable model is shown and marked.")
        add("")
        add(_table(
            ["Category", "Pick", "Success Rate", "Cost / Run",
             "Models Clearing Bar"],
            ["l", "l", "r", "r", "r"],
            [[p["category"],
              p["label"] if p["clears_bar"] else f"{p['label']} (none clear the bar)",
              p["success_rate_percent"], _usd(p["cost_per_run_usd"]),
              p["eligible_models"]]
             for p in picks],
        ))
        add("")

    add("## Per-Model Results")
    add("")
    add(_table(
        ["Model", "Provider", "Successful Runs", "Failed Runs", "Total Runs",
         "Success Rate", "Perfect Samples", "Samples With Failures"],
        ["l", "l", "r", "r", "r", "r", "r", "r"],
        [[_label(m), m["provider"], m["successful_runs"], m["failed_runs"],
          m["total_runs"], m["success_rate_percent"],
          m["samples_passed_all_30_runs"],
          m["samples_with_at_least_one_failure"]]
         for m in results["model_summaries"]],
    ))
    add("")

    add("## Per-Category Results")
    add("")
    add(_table(
        ["Model", "Provider", "Category", "Successful Runs", "Failed Runs",
         "Total Runs", "Success Rate"],
        ["l", "l", "l", "r", "r", "r", "r"],
        [[_label(c), c["provider"], c["category"], c["successful_runs"],
          c["failed_runs"], c["total_runs"], c["success_rate_percent"]]
         for c in results["category_summaries"]],
    ))
    add("")

    add("## Per-Sample Results")
    add("")
    add(_table(
        ["Model", "Provider", "Sample ID", "Category", "Difficulty",
         "Successful Runs", "Failed Runs", "Total Runs", "Success Rate"],
        ["l", "l", "l", "l", "l", "r", "r", "r", "r"],
        [[_label(p), p["provider"], p["sample_id"], p["category"],
          p["difficulty"], p["successful_runs"], p["failed_runs"],
          p["total_runs"], p["success_rate_percent"]]
         for p in results["sample_summaries"]],
    ))
    add("")

    add("## Failure Breakdown")
    add("")
    failure_rows: list[list[Any]] = []
    for m in results["model_summaries"]:
        for failure_value, field in _FAILURE_COUNT_FIELDS.items():
            count = m.get(field, 0)
            if count:
                failure_rows.append([_label(m), m["provider"],
                                     failure_value, count])
    if failure_rows:
        add(_table(["Model", "Provider", "Failure Type", "Count"],
                   ["l", "l", "l", "r"], failure_rows))
    else:
        add("_No failures recorded._")
    add("")

    add("## Most Commonly Missed Tools")
    add("")
    missed = results["failure_statistics"]["most_common_missing_tools"]
    if missed:
        add(_table(["Tool Name", "Count"], ["l", "r"],
                   [[m["tool_name"], m["count"]] for m in missed]))
    else:
        add("_No missing tool calls recorded._")
    add("")

    add("## Most Common Argument Errors")
    add("")
    wrong_args = results["failure_statistics"]["most_common_wrong_arguments"]
    if wrong_args:
        add(_table(["Argument Path", "Count"], ["l", "r"],
                   [[a["argument_path"], a["count"]] for a in wrong_args]))
    else:
        add("_No argument errors recorded._")
    add("")

    schema_violations = results["failure_statistics"].get(
        "most_common_schema_violations") or []
    if schema_violations:
        add("## Most Common Schema Violations")
        add("")
        add(_table(["Argument Path", "Count"], ["l", "r"],
                   [[v["argument_path"], v["count"]] for v in schema_violations]))
        add("")

    add("## Model Configurations (secrets redacted)")
    add("")
    add(_table(
        ["Model", "Provider", "Protocol", "API Key Env", "Endpoint",
         "Deployment", "Extra Params"],
        ["l", "l", "l", "l", "l", "l", "l"],
        [[mc.get("label") or mc.get("name", ""), mc.get("provider", ""),
          mc.get("protocol") or "native",
          mc.get("api_key_env") or "—",
          mc.get("endpoint") or (f"${mc['endpoint_env']}" if mc.get("endpoint_env") else "—"),
          mc.get("deployment") or "—",
          json.dumps(mc.get("api_params") or {}, sort_keys=True)]
         for mc in results.get("model_configs", [])],
    ))
    add("")

    add("## Reproducibility")
    add("")
    add(f"- Config hash: `{s['config_hash']}`")
    add(f"- Sample hash: `{s['sample_hash']}`")
    add(f"- Git commit: {s.get('git_commit') or 'unavailable'}")
    add(f"- Python version: {s['python_version']}")
    add(f"- Platform: {s['platform']}")
    add(f"- Seed: {s['seed']}")
    add(f"- Started UTC: {s['started_utc']}")
    add(f"- Finished UTC: {s['finished_utc']}")
    add(f"- Benchmark version: {s['benchmark_version']}")
    add("")
    add("Model, sample, and run ordering, scoring, inputs, the config "
        "snapshot, sample hash, and output formats are deterministic and "
        "reproducible. Provider APIs may still produce nondeterministic "
        "outputs even at temperature 0, so per-run results can vary between "
        "executions.")
    add("")

    add("## Appendix: Failed Runs")
    add("")
    failed = [run for run in results["runs"] if not run["success"]]
    if not failed:
        add("_No failed runs._")
    for run in failed:
        add(f"### {run['sample_id']} — {run.get('label') or run['model']} "
            f"({run['provider']}) run {run['run_index']}")
        add("")
        add(f"- Failure type: {run.get('failure_type') or 'n/a'}")
        add(f"- Scorer reason: {_md_escape(run['scoring_result']['reason'])}")
        if run.get("error"):
            add(f"- Provider error: {_md_escape(run['error'])}")
        add("")
        steps = run.get("steps") or []
        if len(steps) > 1:
            add("Episode:")
            add("")
            for i, step in enumerate(steps, start=1):
                calls = ", ".join(c["name"] for c in step.get("tool_calls") or []) or "—"
                envelope = f" [{step['envelope']}]" if step.get("envelope") else ""
                turn = f"turn {step['turn']}, " if step.get("turn") else ""
                excerpt = _md_escape((step.get("text") or "")[:160])
                add(f"{i}. {turn}calls: {calls}{envelope}"
                    + (f" — `{excerpt}`" if excerpt else ""))
            add("")
        if run.get("expected_state") is not None:
            add("Expected world:")
            add("")
            add("```json")
            add(json.dumps(run["expected_state"], indent=2, ensure_ascii=False))
            add("```")
            add("")
            add("Actual world:")
            add("")
            add("```json")
            add(json.dumps(run.get("world_state"), indent=2, ensure_ascii=False,
                           default=str))
            add("```")
            add("")
            add("Tool calls (recorded, not scored):")
        else:
            add("Expected tool calls:")
            add("")
            add("```json")
            add(json.dumps(run["expected_tool_calls"], indent=2,
                           ensure_ascii=False))
            add("```")
            add("")
            add("Actual tool calls:")
        add("")
        add("```json")
        add(json.dumps(run["actual_tool_calls"], indent=2,
                       ensure_ascii=False))
        add("```")
        add("")
        if run.get("text_output"):
            add("Text output:")
            add("")
            add("```text")
            add(str(run["text_output"]))
            add("```")
            add("")
        if run.get("raw_model_output") is not None:
            add("<details><summary>Raw model output</summary>")
            add("")
            add("```json")
            add(json.dumps(run["raw_model_output"], indent=2,
                           ensure_ascii=False, default=str))
            add("```")
            add("")
            add("</details>")
            add("")
    return "\n".join(lines).rstrip() + "\n"
