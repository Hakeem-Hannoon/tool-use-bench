# Report and Output Schemas

This document specifies the normalized structure of every file the benchmark
writes. The Markdown report is generated from `outputs/results.json` — never
from a separate calculation path — so the two can never disagree.

## `outputs/results.json`

Top-level keys, in order:

| Key | Type | Description |
|---|---|---|
| `summary` | object | Run-level metadata; always appears before runs. |
| `model_summaries` | array | One object per model (config order). |
| `category_summaries` | array | One object per model × category. |
| `sample_summaries` | array | One object per model × sample. |
| `failure_statistics` | object | Cross-model failure aggregations. |
| `model_configs` | array | Redacted model configs (env var names only). |
| `runs` | array | Every run, deterministically ordered. |

### `summary`

```json
{
  "benchmark_name": "tool-call-bench",
  "benchmark_version": "0.1.0",
  "generated_utc": "2026-07-04T00:00:00Z",
  "started_utc": "2026-07-04T00:00:00Z",
  "finished_utc": "2026-07-04T00:30:00Z",
  "config_path": "examples/config.example.json",
  "config_hash": "sha256:…",
  "sample_file": "benchmark_samples/full_benchmark.json",
  "sample_hash": "sha256:…",
  "git_commit": "abc1234",
  "python_version": "3.11.9",
  "platform": "macOS-…",
  "seed": 12345,
  "runs_per_sample": 5,
  "sample_count": 67,
  "model_count": 3,
  "total_runs_per_model": 335,
  "total_runs_all_models": 1005
}
```

### `model_summaries[]`

`model`, `provider`, `successful_runs`, `failed_runs`, `total_runs`,
`success_rate` (float, `successful_runs / total_runs`),
`success_rate_percent` (string, 2 decimals), `samples_passed_all_30_runs`
(the key name is fixed for schema stability; it counts samples that passed
**all** `runs_per_sample` runs, whatever that value is — 5 by default),
`samples_with_at_least_one_failure`, `average_success_rate_per_sample`,
and one counter per primary failure type: `missing_tool_call_failures`,
`wrong_tool_name_failures`, `wrong_argument_failures`,
`extra_tool_call_failures`, `ordering_failures`, `provider_error_failures`,
`timeout_failures`. Each failed run is counted under exactly one type
(precedence: timeout → provider error → missing → wrong name → wrong
argument → extra → ordering), so the seven counters sum to `failed_runs`.

### `category_summaries[]`

`model`, `provider`, `category`, `successful_runs`, `failed_runs`,
`total_runs`, `success_rate`, `success_rate_percent`. Success rate is
`successful_runs_for_that_category / total_runs_for_that_category`.

### `sample_summaries[]`

`model`, `provider`, `sample_id`, `category`, `difficulty`,
`successful_runs`, `failed_runs`, `total_runs`, `success_rate`
(`successful_runs_for_that_sample / runs_per_sample`),
`success_rate_percent`.

### `failure_statistics`

```json
{
  "most_common_missing_tools": [{"tool_name": "…", "count": 18}],
  "most_common_wrong_arguments": [{"argument_path": "…", "count": 12}],
  "failure_type_counts": {
    "missing_tool_call": 0, "wrong_tool_name": 0, "wrong_argument": 0,
    "extra_tool_call": 0, "ordering_error": 0, "provider_error": 0,
    "timeout": 0
  }
}
```

### `runs[]`

```json
{
  "model": "gpt-4.1",
  "provider": "openai",
  "sample_id": "0001_weather_current",
  "run_index": 0,
  "success": true,
  "expected_tool_calls": [{"name": "get_weather", "arguments": {"location": "Toronto, Canada", "unit": "celsius"}}],
  "actual_tool_calls": [{"name": "get_weather", "arguments": {"location": "Toronto, Canada", "unit": "celsius"}}],
  "raw_model_output": {},
  "text_output": "",
  "scoring_result": {
    "success": true,
    "missing_tool_calls": [],
    "wrong_tool_calls": [],
    "argument_errors": [],
    "extra_tool_calls": [],
    "ordering_errors": [],
    "reason": "Passed."
  },
  "latency_ms": 1234,
  "error": null,
  "timed_out": false,
  "failure_type": null
}
```

`raw_model_output` is `null` when `save_raw_responses` is `false`.
Run order is deterministic: models in config order, then sample id, then
run index.

## Other JSON outputs

| File | Contents |
|---|---|
| `outputs/model_summaries.json` | `summary` + `model_summaries` |
| `outputs/sample_summaries.json` | `summary` + `sample_summaries` |
| `outputs/category_summaries.json` | `summary` + `category_summaries` |
| `outputs/failures.json` | `summary` + `failure_statistics` + only the failed runs (expected calls, actual calls, raw output, text output, scorer reason) |
| `outputs/checkpoints/checkpoint.json` | Run identity (benchmark version, config hash, sample hash) + completed run keys |
| `outputs/checkpoints/records.jsonl` | One completed run record per line (append-only) |
| `outputs/run_config.snapshot.json` | Redacted copy of the exact config used + reproducibility metadata |

## `reports/tool_call_report.md`

Sections, in order: title; generation timestamps, git commit, version,
config path/hash, sample hash, counts; **Overall Ranking** (sorted by
success rate descending); **Per-Model Results**; **Per-Category Results**;
**Per-Sample Results**; **Failure Breakdown**; **Most Commonly Missed
Tools**; **Most Common Argument Errors**; **Model Configurations (secrets
redacted)**; **Reproducibility**; **Appendix: Failed Runs** (every failed
run with expected calls, actual calls, raw output when saved, text output,
and the scorer's reason). See `analysis/example_report.md` for a rendered
example.
