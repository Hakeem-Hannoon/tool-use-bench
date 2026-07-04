# Tool Call Benchmark Report

Generated UTC: 2026-07-04T00:00:00Z  
Generated local: 2026-07-03T20:00:00-04:00  
Git commit: abc1234  
Benchmark version: 0.1.0  
Config file: examples/config.example.json  
Config hash: `sha256:1f0e…`  
Sample file: benchmark_samples/full_benchmark.json  
Sample hash: `sha256:9c2a…`  
Models: 2  
Samples: 67  
Runs per sample: 5  
Total runs per model: 335  
Total runs overall: 670  

## Overall Ranking

| Rank | Model | Provider | Successful Runs | Total Runs | Success Rate |
|---:|---|---|---:|---:|---:|
| 1 | gpt-4.1 | openai | 316 | 335 | 94.33% |
| 2 | claude-sonnet-4 | anthropic | 312 | 335 | 93.13% |

## Per-Model Results

| Model | Provider | Successful Runs | Failed Runs | Total Runs | Success Rate | Perfect Samples | Samples With Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1 | openai | 316 | 19 | 335 | 94.33% | 52 | 15 |
| claude-sonnet-4 | anthropic | 312 | 23 | 335 | 93.13% | 49 | 18 |

## Per-Category Results

| Model | Provider | Category | Successful Runs | Failed Runs | Total Runs | Success Rate |
|---|---|---|---:|---:|---:|---:|
| gpt-4.1 | openai | argument_precision | 44 | 6 | 50 | 88.00% |
| gpt-4.1 | openai | multi_tool | 45 | 5 | 50 | 90.00% |
| gpt-4.1 | openai | no_tool_decoy | 38 | 2 | 40 | 95.00% |
| gpt-4.1 | openai | ordering_required | 27 | 3 | 30 | 90.00% |
| gpt-4.1 | openai | parallel_tools | 39 | 1 | 40 | 97.50% |
| gpt-4.1 | openai | single_tool | 98 | 2 | 100 | 98.00% |
| gpt-4.1 | openai | tool_choice_under_ambiguity | 25 | 0 | 25 | 100.00% |

*(…same seven rows per additional model; truncated in this example…)*

## Per-Sample Results

| Model | Provider | Sample ID | Category | Difficulty | Successful Runs | Failed Runs | Total Runs | Success Rate |
|---|---|---|---|---|---:|---:|---:|---:|
| gpt-4.1 | openai | 0001_weather_current | single_tool | easy | 5 | 0 | 5 | 100.00% |
| gpt-4.1 | openai | 0002_calendar_search | single_tool | easy | 4 | 1 | 5 | 80.00% |

*(…one row per model × sample; truncated in this example…)*

## Failure Breakdown

| Model | Provider | Failure Type | Count |
|---|---|---|---:|
| gpt-4.1 | openai | missing_tool_call | 7 |
| gpt-4.1 | openai | wrong_tool_name | 2 |
| gpt-4.1 | openai | wrong_argument | 6 |
| gpt-4.1 | openai | extra_tool_call | 2 |
| gpt-4.1 | openai | ordering_error | 2 |

## Most Commonly Missed Tools

| Tool Name | Count |
|---|---:|
| create_calendar_event | 4 |
| create_reminder | 2 |

## Most Common Argument Errors

| Argument Path | Count |
|---|---:|
| location | 3 |
| changes.start_time | 2 |

## Model Configurations (secrets redacted)

| Model | Provider | API Key Env | Endpoint | Deployment | Extra Params |
|---|---|---|---|---|---|
| gpt-4.1 | openai | OPENAI_API_KEY | — | — | {} |
| claude-sonnet-4 | anthropic | ANTHROPIC_API_KEY | — | — | {} |

## Reproducibility

- Config hash: `sha256:1f0e…`
- Sample hash: `sha256:9c2a…`
- Git commit: abc1234
- Python version: 3.11.9
- Platform: macOS-15.5-arm64-arm-64bit
- Seed: 12345
- Started UTC: 2026-07-04T00:00:00Z
- Finished UTC: 2026-07-04T00:30:00Z
- Benchmark version: 0.1.0

Model, sample, and run ordering, scoring, inputs, the config snapshot,
sample hash, and output formats are deterministic and reproducible.
Provider APIs may still produce nondeterministic outputs even at
temperature 0, so per-run results can vary between executions.

## Appendix: Failed Runs

### 0002_calendar_search — gpt-4.1 (openai) run 3

- Failure type: wrong_argument
- Scorer reason: Failed — argument errors: search_calendar_events.start_date (extra_argument).

Expected tool calls:

```json
[{"name": "search_calendar_events", "arguments": {"query": "dentist"}}]
```

Actual tool calls:

```json
[{"name": "search_calendar_events", "arguments": {"query": "dentist", "start_date": "2026-01-01"}}]
```

*(…one section per failed run…)*
