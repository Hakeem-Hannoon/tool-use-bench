# tool-call-bench

A benchmark that measures how reliably LLMs **select and emit the correct
tool calls**. It runs a fixed set of 67 tool-calling samples against multiple
models, repeats each sample 5 times per model by default, scores every run
deterministically (no LLM judge), and writes machine-readable JSON results
plus a normalized Markdown report.

## Why this exists

This benchmark was built to answer a production question, not an academic
one: **which model should power each agentic workload, and at what cost?**
It drives model selection for my products — most notably
[Spotter](https://spotter.com), an agent-heavy training companion that keeps
track of your gym progress. Nearly everything Spotter does rides on tool
calls: logging sets, scheduling sessions, querying training history,
nudging you at the right moment. At that level of dependence, a model that
fumbles arguments 3% of the time isn't 3% worse — it's a broken feature.

tool-call-bench turns that choice into data. Because every run is scored
deterministically and broken down by category (single calls, parallel
fan-outs, argument precision, decoy resistance, ordering), each candidate
model gets a reliability profile that maps directly onto real app
workloads. Read that profile against the provider's price per call and the
cost/capability trade-off stops being a matter of taste: the cheapest model
that clears the reliability bar for a given workload wins that workload.

## What it measures

- Did the model call **all required tools**?
- Did it call them with **exactly the right arguments** (values, types,
  required keys, no extras)?
- Did it respect **required ordering** when order matters?
- Did it avoid **extra tool calls** when extras are disallowed?
- Did it correctly **decline to call any tool** on decoy prompts?
- Did it emit **real provider-native tool calls** (not JSON pasted into
  text)?

## What it does NOT measure

- Natural-language answer quality, style, or helpfulness.
- Tool *results* — tools are mock schemas and are never executed.
- Multi-turn behavior — every sample is a single turn.

A run is **successful only if the model emits all required tool calls
correctly**. Any one of the following fails the entire run:

- a required tool call is missing,
- the wrong tool is called,
- an argument value is wrong or has the wrong JSON type (`"5"` ≠ `5`),
- a required argument is omitted,
- an unexpected extra argument is present (when exact arguments required),
- an extra tool call is made while `allow_extra_tool_calls` is false,
- required ordering is violated,
- the model returns only text instead of real tool calls,
- the provider call times out, or
- the provider returns an execution error after retries.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_ORG/tool-call-bench
cd tool-call-bench
uv sync
```

## Set API keys

Copy `.env.example` to `.env` and fill in the keys for the providers you
want to benchmark (or export them in your shell):

```bash
cp .env.example .env
# edit .env
```

Secrets are read **only** from environment variables at runtime. They are
never written to any output file — config snapshots store environment
variable *names*, never values.

> ⚠️ **Never commit `.env` or API keys.** `.env` is gitignored; keep it
> that way.

## Quick start

```bash
# 1. Generate the 67 samples + the combined benchmark file
uv run benchmark generate-samples

# 2. Validate them (also runs automatically before any benchmark run)
uv run benchmark validate-samples benchmark_samples/full_benchmark.json

# 3. Run a tiny smoke test first: copy the example config, keep ONE model,
#    and set "runs_per_sample": 1
cp examples/config.example.json examples/config.tiny.json
#    ...edit examples/config.tiny.json...
uv run benchmark run examples/config.tiny.json

# 4. Full benchmark
uv run benchmark run examples/config.example.json

# 5. Re-generate the report or print a terminal summary from saved results
uv run benchmark report outputs/results.json --out reports/tool_call_report.md
uv run benchmark summarize outputs/results.json
```

> 💰 **Cost warning:** the full benchmark is
> `67 samples × 5 runs × number of models = 335 calls per model`
> (`runs_per_sample: 30` would mean 2,010 calls per model). Six configured
> models at the default means 2,010 API calls. Start with a tiny config.

## How scoring works

Scoring is fully deterministic — plain data comparison, no LLM judge.

1. **Tool call extraction.** Only provider-native tool/function calls count:
   OpenAI Chat Completions `tool_calls` (and Responses-API `function_call`
   items), Azure OpenAI tool calls, Anthropic `tool_use` blocks,
   Gemini/Vertex `function_call` parts, and OpenAI-style calls from
   OpenRouter/OpenAI-compatible endpoints. If a model prints JSON that
   *looks* like a tool call in its text, the run fails by default. Setting
   `"allow_text_tool_call_fallback": true` lets the benchmark parse valid
   JSON tool calls out of text — the default is strict.
2. **Matching.** Expected and actual calls are matched by tool name and
   argument equality. With `require_order: true` the expected calls must
   appear in order; otherwise any order is accepted.
3. **Argument comparison.** Arguments compare as JSON values. Strings are
   trimmed of surrounding whitespace when `trim_string_whitespace` is true.
   Types are strict by default: `"5"` (string) never equals `5` (number)
   unless the sample sets `allow_argument_coercion: true`. Integers and
   floats are both JSON numbers, so `5` equals `5.0`; booleans are never
   numbers. With `require_exact_arguments: true` the argument object must
   match key-for-key (missing and extra keys both fail); with `false`,
   expected arguments must be a subset.
4. **Verdict.** Each run yields a scoring result:

```json
{
  "success": true,
  "missing_tool_calls": [],
  "wrong_tool_calls": [],
  "argument_errors": [],
  "extra_tool_calls": [],
  "ordering_errors": [],
  "reason": "Passed."
}
```

Each **failed** run is classified under exactly one primary failure type
(precedence: timeout → provider error → missing call → wrong name → wrong
argument → extra call → ordering) so summary counters always add up.

### Success-rate math

- Model total: `total_successful_runs / total_runs`
- Per sample: `successful_runs_for_that_sample / runs_per_sample` (5 by default)
- Per category: `successful_runs_for_that_category / total_runs_for_that_category`

## The dataset

Exactly **67 samples**, each a standalone JSON file under
`benchmark_samples/tool_calling/`, combined into
`benchmark_samples/full_benchmark.json`. Categories:

| Category | Count | Tests |
|---|---:|---|
| `single_tool` | 20 | One required call with precise arguments |
| `multi_tool` | 10 | Several different tools in one turn (incl. `0067_multi_tool_complex_workflow`) |
| `parallel_tools` | 8 | Independent same-tool fan-out calls |
| `no_tool_decoy` | 8 | No call expected; any call fails |
| `argument_precision` | 10 | Exact dates, units, IDs, recipients, enums, nested objects, array order |
| `ordering_required` | 6 | Calls must appear in a required order |
| `tool_choice_under_ambiguity` | 5 | Similar/competing tools; the right one must be chosen |

Tools covered: weather (current + forecast), calendar search/create/update,
email search/draft/send, contacts, web search, calculator, currency and
unit conversion, current time, file search, document retrieval, spreadsheet
update, GitHub issues and PRs, Slack channel + DM mocks, reminders,
geocoding, restaurants, flight and hotel search mocks, code execution mock,
database query mock, CRM update mock, refund lookup mock, and support
tickets. All are **mock schemas** — nothing is executed.

Sample schema (see `benchmark_samples/tool_calling/0001_weather_current.json`):

```json
{
  "id": "0001_weather_current",
  "category": "single_tool",
  "difficulty": "easy",
  "prompt": "What is the current weather in Toronto, Canada? Use Celsius.",
  "system_prompt": "",
  "tools": [{ "type": "function", "function": { "name": "get_weather", "…": "…" } }],
  "expected_tool_calls": [{ "name": "get_weather", "arguments": { "location": "Toronto, Canada", "unit": "celsius" } }],
  "scoring": {
    "require_exact_tool_names": true,
    "require_exact_arguments": true,
    "allow_extra_tool_calls": false,
    "require_order": true,
    "allow_argument_coercion": false,
    "trim_string_whitespace": true
  },
  "notes": "…"
}
```

When a sample's `system_prompt` is empty, the default system prompt from
`prompts/default_system_prompt.txt` is used.

## Configuration

Runs are driven by a JSON config (see `examples/`):

| Field | Default | Meaning |
|---|---|---|
| `benchmark` | `benchmark_samples/full_benchmark.json` | Sample set to run |
| `output` | `outputs/results.json` | Main results file |
| `report` | `reports/tool_call_report.md` | Markdown report path |
| `runs_per_sample` | `5` | Repetitions per model per sample |
| `concurrency` | `5` | Max in-flight API calls |
| `temperature` | `0` | Sampling temperature (default 0) |
| `max_retries` | `3` | Retries for transient provider errors |
| `timeout_seconds` | `90` | Per-call timeout |
| `allow_text_tool_call_fallback` | `false` | Parse JSON tool calls from text (strict off by default) |
| `resume` | `true` | Resume from checkpoint |
| `force` | `false` | Discard checkpoint, rerun everything |
| `seed` | `12345` | Recorded for reproducibility |
| `save_raw_responses` | `true` | Store raw provider responses in results |
| `models` | — | List of model/provider entries |

Per-model `api_params` are merged into every request for that model
(overriding computed values); a JSON `null` value **removes** a parameter —
useful for models that reject `temperature`.

## Provider setup

Supported providers: `openai`, `azure_openai`, `anthropic`, `gemini`,
`vertexai`, `openrouter`, `openai_compatible`.

**OpenAI**

```json
{ "name": "gpt-4.1", "provider": "openai", "api_key_env": "OPENAI_API_KEY", "endpoint": null, "api_params": {} }
```

**Azure OpenAI** — routes by `deployment`; `endpoint` and `api_version`
required:

```json
{ "name": "gpt-4.1", "provider": "azure_openai", "api_key_env": "AZURE_OPENAI_API_KEY",
  "endpoint": "https://YOUR_RESOURCE.openai.azure.com", "deployment": "gpt-4.1",
  "api_version": "2025-01-01-preview", "api_params": {} }
```

**Anthropic** — `max_tokens` defaults to 4096; override via
`"api_params": {"max_tokens": 8192}`. For models that reject sampling
parameters, drop temperature with `"api_params": {"temperature": null}`:

```json
{ "name": "claude-sonnet-4", "provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "endpoint": null, "api_params": {} }
```

**Gemini (Developer API)**

```json
{ "name": "gemini-2.5-pro", "provider": "gemini", "api_key_env": "GEMINI_API_KEY", "endpoint": null, "api_params": {} }
```

**Vertex AI** — authenticates with Application Default Credentials; set
`VERTEXAI_PROJECT_ID`, `VERTEXAI_LOCATION`, and (optionally)
`GOOGLE_APPLICATION_CREDENTIALS` pointing at a service-account JSON:

```json
{ "name": "gemini-2.5-pro", "provider": "vertexai", "api_key_env": null, "endpoint": null,
  "project_id_env": "VERTEXAI_PROJECT_ID", "location_env": "VERTEXAI_LOCATION",
  "service_account_path_env": "GOOGLE_APPLICATION_CREDENTIALS", "api_params": {} }
```

**OpenRouter**

```json
{ "name": "anthropic/claude-sonnet-4", "provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY", "endpoint": null, "api_params": {} }
```

**OpenAI-compatible** — any endpoint speaking the OpenAI chat/tools API
(DeepSeek, Groq, Together, Fireworks, Mistral, Cerebras, Perplexity, xAI,
local gateways, …); `endpoint` is required:

```json
{ "name": "deepseek-chat", "provider": "openai_compatible", "api_key_env": "DEEPSEEK_API_KEY",
  "endpoint": "https://api.deepseek.com/v1", "api_params": {} }
```

## Checkpointing and resume

A checkpoint is saved after **every** completed model/sample/run
combination:

- `outputs/checkpoints/records.jsonl` — one completed run per line
  (append-only, crash-safe).
- `outputs/checkpoints/checkpoint.json` — run identity + completed run keys.

If a run is interrupted, rerunning the same command resumes and skips
completed runs. Checkpoint keys include the benchmark version, config hash,
sample hash, provider, model, sample id, and run index — so changing the
config or the samples never reuses stale results. Completed runs are never
silently overwritten: with `--no-resume` and an existing checkpoint the run
aborts with instructions; `--force` explicitly discards the checkpoint and
reruns everything.

```bash
uv run benchmark run examples/config.example.json --resume     # default
uv run benchmark run examples/config.example.json --no-resume  # refuse to reuse
uv run benchmark run examples/config.example.json --force      # rerun all
```

## Reproducibility

Everything the benchmark controls is deterministic:

- model, sample, and run ordering are fixed (config order → sample id →
  run index);
- scoring is pure data comparison — no LLM judge;
- default temperature is `0`;
- the exact redacted config is snapshotted to
  `outputs/run_config.snapshot.json` together with the config hash, sample
  hash, git commit (when available), Python version, package version,
  OS/platform, seed, and UTC start/finish times;
- results embed the same metadata in `summary`.

**Caveat:** provider APIs may still produce nondeterministic outputs even
at temperature 0 (backend changes, sampling implementation details,
hardware). The benchmark inputs, ordering, scoring, config snapshot, sample
hash, and output formats are reproducible; individual model outputs may not
be.

## Output files

| File | Contents |
|---|---|
| `outputs/results.json` | Complete results: `summary` first, then model/category/sample summaries, failure statistics, redacted model configs, and every run |
| `outputs/model_summaries.json` | One summary object per model |
| `outputs/sample_summaries.json` | Per-model, per-sample performance |
| `outputs/category_summaries.json` | Per-model, per-category performance |
| `outputs/failures.json` | Only failed runs (expected vs actual calls, raw output, text, scorer reason) |
| `outputs/checkpoints/checkpoint.json` | Resumable checkpoint state |
| `outputs/run_config.snapshot.json` | Redacted copy of the exact config used |
| `reports/tool_call_report.md` | Markdown report generated from `results.json` |

Full field-by-field schemas: [`analysis/report_schema.md`](analysis/report_schema.md).
A rendered example report: [`analysis/example_report.md`](analysis/example_report.md).

## The Markdown report

`reports/tool_call_report.md` is generated **from `outputs/results.json`**
(single calculation path — the report and the JSON cannot disagree). It
contains generation timestamps, git commit, version, config path + hash,
sample hash, run counts, an overall ranking sorted by success rate,
per-model / per-category / per-sample tables, a failure breakdown, the most
commonly missed tools, the most common argument errors, the redacted model
configuration summary, a reproducibility section, and an appendix with
every failed run (expected calls, actual calls, raw output when saved, text
output, and the scorer's reason).

## Tests

```bash
uv run pytest
```

Provider SDKs are never called in tests — provider normalization is tested
against recorded response shapes, and the checkpoint/runner tests use a
fake in-process provider.

## Project layout

```text
benchmark_samples/   67 sample files + full_benchmark.json
examples/            example run configs
prompts/             default system prompt
analysis/            output schema docs + example report
src/benchmark/       package: cli, config, schema, runner, scoring,
                     checkpoint, reporting, samples, tool_schemas,
                     providers/, utils/
tests/               unit tests
outputs/, reports/   generated artifacts (gitignored)
```

## Limitations & provider notes

- Exact-match scoring is intentionally strict: samples pin free-text
  arguments in quotes, but a model that rephrases a pinned string fails.
  That strictness is the benchmark's point — read per-sample results with
  it in mind.
- `temperature: 0` is sent by default; some newer models reject sampling
  parameters — drop them per model with `"api_params": {"temperature": null}`.
- Anthropic requires `max_tokens` (default 4096 here); very long parallel
  call sets may need a higher value via `api_params`.
- Gemini/Vertex function declarations don't accept `additionalProperties`;
  the adapter strips it when converting schemas (samples still validate the
  strict schema shape).
- Vertex AI uses Application Default Credentials, not an API key.
- OpenRouter normalizes many vendors to the OpenAI shape, but vendor quirks
  upstream of OpenRouter can still surface as provider errors — these are
  tracked separately as `provider_error_failures`.
