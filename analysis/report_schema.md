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
| `workload_picks` | array | Per category, the cheapest model at or above `reliability_bar`. |
| `failure_statistics` | object | Cross-model failure aggregations. |
| `model_configs` | array | Redacted model configs (env var names only). |
| `runs` | array | Every run, deterministically ordered. |

### `summary`

```json
{
  "benchmark_name": "tool-call-bench",
  "benchmark_version": "0.2.0",
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
  "total_runs_all_models": 1005,
  "reliability_bar": 0.95
}
```

### `model_summaries[]`

One per model **entry** (config order), grouped by `model_key`, so the same
model under two protocols or labels stays separate.

`model`, `provider`, `model_key`, `label`, `protocol`, `successful_runs`,
`failed_runs`, `total_runs`,
`success_rate` (float, `successful_runs / total_runs`),
`success_rate_percent` (string, 2 decimals), `samples_passed_all_30_runs`
(the key name is fixed for schema stability; it counts samples that passed
**all** `runs_per_sample` runs, whatever that value is — 5 by default),
`samples_with_at_least_one_failure`, `average_success_rate_per_sample`,
one counter per primary failure type: `missing_tool_call_failures`,
`wrong_tool_name_failures`, `schema_violation_failures`,
`wrong_argument_failures`, `extra_tool_call_failures`, `ordering_failures`,
`protocol_error_failures`, `provider_error_failures`, `timeout_failures`;
and cost/latency fields: `avg_input_tokens`, `avg_cached_input_tokens`,
`avg_output_tokens` (per run, all steps), `total_cost_usd`,
`cost_per_run_usd`, `cost_per_successful_run_usd` (`null` without
`pricing`), `latency_p50_ms`, `latency_p95_ms` (nearest rank, provider
errors and timeouts excluded), `avg_steps`, `recoveries`. Each failed run
is counted under exactly one type (precedence: timeout → provider error →
protocol error → missing → wrong name → schema violation → wrong argument →
extra → ordering), so the nine counters sum to `failed_runs`.

### `category_summaries[]`

`model`, `provider`, `model_key`, `label`, `protocol`, `category`,
`successful_runs`, `failed_runs`, `total_runs`, `success_rate`,
`success_rate_percent`, `cost_per_run_usd`, `latency_p50_ms`. Success rate is
`successful_runs_for_that_category / total_runs_for_that_category`.

### `workload_picks[]`

`category`, `clears_bar` (false when no model reached `reliability_bar`; the
most reliable model is shown instead), `model_key`, `label`,
`success_rate`, `success_rate_percent`, `cost_per_run_usd`,
`eligible_models`.

### `sample_summaries[]`

`model`, `provider`, `model_key`, `label`, `protocol`, `sample_id`, `category`, `difficulty`,
`successful_runs`, `failed_runs`, `total_runs`, `success_rate`
(`successful_runs_for_that_sample / runs_per_sample`),
`success_rate_percent`.

### `failure_statistics`

```json
{
  "most_common_missing_tools": [{"tool_name": "…", "count": 18}],
  "most_common_wrong_arguments": [{"argument_path": "tool.path", "count": 12}],
  "most_common_schema_violations": [{"argument_path": "tool.path", "count": 3}],
  "failure_type_counts": {
    "missing_tool_call": 0, "wrong_tool_name": 0, "schema_violation": 0,
    "wrong_argument": 0, "extra_tool_call": 0, "ordering_error": 0,
    "protocol_error": 0, "provider_error": 0, "timeout": 0
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
  "failure_type": null,
  "model_key": "openai/gpt-4.1",
  "label": "gpt-4.1",
  "protocol": "native",
  "usage": {"input_tokens": 512, "output_tokens": 24, "cached_input_tokens": 0},
  "cost_usd": 0.0012,
  "step_count": 1,
  "recoveries": 0,
  "steps": [{"tool_calls": [], "text": "", "usage": {}, "latency_ms": 1234,
             "envelope": null, "raw_response": {}}]
}
```

Expected calls also carry `alternatives` (empty unless the sample lists
them), and `scoring_result` carries `schema_errors` and `protocol_error`.
`raw_model_output` (single-step runs) and each step's `raw_response` are
`null` when `save_raw_responses` is `false`. `steps[].envelope` is set
under the router protocol: `tool`, `reply`, `prose`, `invalid_json`,
`unknown_tool`, or `empty`.
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
success rate descending); **Cost & Latency**; **Workload Picks**;
**Per-Model Results**; **Per-Category Results**;
**Per-Sample Results**; **Failure Breakdown**; **Most Commonly Missed
Tools**; **Most Common Argument Errors**; **Most Common Schema
Violations** (when any); **Model Configurations (secrets redacted)**; **Reproducibility**; **Appendix: Failed Runs** (every failed
run with expected calls, actual calls, a step trace for multi-step
episodes, raw output when saved, text output, and the scorer's reason). See `analysis/example_report.md` for a rendered
example.
