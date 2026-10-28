# tool-call-bench

A benchmark that measures how reliably LLMs **select and emit the correct
tool calls**, and what each correct call costs. It runs fixed sample suites
against multiple models, repeats each sample 5 times per model by default,
scores every run with deterministic data comparison, and writes
machine-readable JSON results plus a normalized Markdown report with
success rates, tokens, cost, latency, and the cheapest model that clears
the reliability bar for each workload.

Two suites ship:

- **generic** — 67 single-turn samples over common mock tools (weather,
  email, calendar, GitHub, …).
- **spotter** — 70 samples built from [Spotter](https://www.spotter-labs.com)'s
  real agent: its 42 production tools with Zod-derived schemas, its real
  system prompts and domain routing, its JSON-router tool protocol, and
  multi-step episodes against a fixture user. Fifteen of them are
  multi-turn dialogues scored on the final world rather than one call trace.

## Why this exists

This benchmark exists to answer one production question: **which model
should power each agentic workload, and at what cost?** It drives model
selection for the things I build — most notably
[Spotter](https://www.spotter-labs.com), an agent-heavy training companion
that keeps track of your gym progress. Nearly everything Spotter does rides on tool
calls: logging sets, scheduling sessions, querying training history,
nudging you at the right moment. At that level of dependence, a model that
fumbles arguments on 3% of calls ships a broken feature to someone every
single day.

tool-call-bench turns that choice into data. Because every run is scored
deterministically and broken down by category (single calls, parallel
fan-outs, argument precision, decoy resistance, ordering — or, in the
Spotter suite, logging, id-grounded edits, nutrition, charts, …), each
candidate model gets a reliability profile that maps directly onto real app
workloads. Every run also records its tokens and, with pricing configured,
its cost, so the report settles the cost/capability trade-off on evidence:
the cheapest model that clears the reliability bar for a given workload
wins that workload.

## What it measures

- Did the model call **all required tools**?
- Did it call them with **exactly the right arguments** (values, types,
  required keys, zero extras)?
- Did it respect **required ordering** when order matters?
- Did it avoid **extra tool calls** when extras are disallowed?
- Did it correctly **decline to call any tool** on decoy prompts?
- Did it emit its calls through the configured **tool protocol**? Under
  `native`, JSON printed inside a text response fails by default; under
  `spotter_router`, the reply must be Spotter's JSON envelope.
- Do the arguments pass the tool's **JSON Schema** (what a server-side
  validator such as Zod would reject), when a sample opts in?
- In **multi-step episodes**, did it act on the data the tools returned —
  e.g. edit a set by the exact id a read gave it?
- In a **conversation**, does the fixture world end in the right state —
  the set weights, the deleted ids, the memories — whichever legal trace
  got there?

## Out of scope

- Natural-language answer quality, style, or helpfulness. Reply text is
  recorded but never scored.
- Real tool execution — tools are mock schemas. Multi-step samples return
  scripted results; scoring looks only at the call objects a model emits.
- Multi-turn conversations outside the Spotter `conversation` category.
  Every other sample is one user message (which may take several model
  steps). The conversation category scores the final world, not reply text.

A run is **successful only if the model emits all required tool calls
correctly**. Any one of the following fails the entire run:

- a required tool call is missing,
- the wrong tool is called,
- an argument value is wrong or has the wrong JSON type (`"5"` ≠ `5`),
- a required argument is omitted,
- an unexpected extra argument is present (when exact arguments required),
- an extra tool call is made while `allow_extra_tool_calls` is false,
- required ordering is violated,
- the model answers in plain text and emits zero tool calls,
- an argument violates the tool's schema (when `validate_arguments_schema`),
- the router protocol never produces a parseable envelope (protocol error),
- the provider call times out, or
- the provider returns an execution error after retries.

A `conversation` sample replaces that call checklist with the world diff.
Timeout, a provider error, and a protocol error still fail it. A trace that
uses different reads or a different grouping passes when the final sets,
deletes, and memories match.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hakeem-Hannoon/tool-use-bench
cd tool-use-bench
uv sync
```

## Set API keys

Copy `.env.example` to `.env` and fill in the keys for the providers you
want to benchmark (or export them in your shell):

```bash
cp .env.example .env
# edit .env
```

Secrets are read **only** from environment variables at runtime, and they
stay out of every output file: config snapshots record just the
environment variable *names*. `.env` is gitignored.

## Quick start

```bash
# 1. Generate both suites (or --suite generic / --suite spotter)
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
> Spotter episodes can take several steps each and carry a 30–50 KB system
> prompt, so budget per step. A conversation sample repeats that loop once
> per user turn. The report's Cost & Latency table shows the real figure
> after a smoke run.

## How scoring works

Scoring is fully deterministic, implemented as plain data comparison.

1. **Tool call extraction.** Only provider-native tool/function calls count:
   OpenAI Chat Completions `tool_calls` (and Responses-API `function_call`
   items), Azure OpenAI tool calls, Anthropic `tool_use` blocks,
   Gemini/Vertex `function_call` parts, and OpenAI-style calls from
   OpenRouter/OpenAI-compatible endpoints. If a model prints JSON that
   *looks* like a tool call in its text, the run fails by default. Setting
   `"allow_text_tool_call_fallback": true` lets the benchmark parse valid
   JSON tool calls out of text — the default is strict.
2. **Matching.** Expected and actual calls are matched by tool name and
   argument equality, using an optimal assignment (never greedy pairing).
   With `require_order: true` the expected calls must appear in order;
   otherwise any order is accepted. An expected call may list
   `alternatives` — other name/argument pairs that satisfy it (any of
   several read tools, or two accepted argument shapes). Calls to tools in
   `allowed_extra_tools` (harmless reads, say) never count as extra.
3. **Argument comparison.** Arguments compare as JSON values. Strings are
   trimmed of surrounding whitespace when `trim_string_whitespace` is true.
   Types are strict by default: `"5"` (string) and `5` (number) count as
   different values unless the sample sets `allow_argument_coercion: true`.
   Integers and floats are both JSON numbers, so `5` equals `5.0`, while
   booleans form their own type. With `require_exact_arguments: true` the
   argument object must
   match key-for-key (missing and extra keys both fail); with `false`,
   expected arguments must be a subset — at every nesting level.
   Expected values may be deterministic **matchers** instead of literals:

   | Matcher | Matches |
   |---|---|
   | `{"$any": true}` | key present, any value |
   | `{"$absent": true}` | key must not be present |
   | `{"$optional": v}` | key may be absent; if present must match `v` |
   | `{"$ci": "text"}` | string equal ignoring case and repeated spaces |
   | `{"$contains": "t"}` / `["a","b"]` | string contains every fragment (ignoring case) |
   | `{"$regex": "…"}` | string fully matches the regex |
   | `{"$one_of": [v1, v2]}` | any alternative (alternatives may be matchers) |
   | `{"$range": [lo, hi]}` | number in range (`null` = unbounded) |
   | `{"$unordered": [v1, v2]}` | array with exactly these items, any order |
   | `{"$includes": [v1, v2]}` | array containing at least these items |

   With `validate_arguments_schema: true`, every emitted call is also
   validated against the offered tool's JSON Schema.
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
(precedence: timeout → provider error → protocol error → world mismatch →
missing call → wrong name → schema violation → wrong argument → extra call →
ordering) so
summary counters always add up.

### Success-rate math

- Model total: `total_successful_runs / total_runs`
- Per sample: `successful_runs_for_that_sample / runs_per_sample` (5 by default)
- Per category: `successful_runs_for_that_category / total_runs_for_that_category`

## Tool protocols and episodes

Each model entry picks a `protocol`:

- **`native`** (default) — the provider's own tool-calling API. Tools are
  sent as `tools`; only native tool-call objects count.
- **`spotter_router`** — Spotter's production contract, ported from
  `apps/web/lib/agent/orchestrator.ts`: no `tools` parameter, the catalog
  lives in the system prompt, the request runs in JSON mode, and each step
  the model returns one object — `{"kind":"tool","tool":…,"arguments":…}`
  or `{"kind":"reply",…}`. Near-miss tool names are canonicalized
  (`search_workout` → `search_workouts`) and invalid JSON or unknown tools
  get production's one-time nudge, counted as a **recovery** in the report.
  An envelope that never parses is a **protocol error**.

Benchmarking the same model under both protocols answers "should we switch
to native tool calling?" with data.

A sample with `max_steps > 1` runs as an **episode**: each non-terminal
call gets the first matching scripted result from `tool_results` (matched
on a subset of its arguments, matchers allowed) and the model continues
until it replies, calls a terminal tool (`terminal_tools`: writes, display
cards), or runs out of steps. With `continue_after_terminal` the model is
told the proposal was created and may make another call (compound
requests). Every call across the episode is scored together, and tokens
are summed across steps. Single-step samples behave exactly as before.

## The Spotter suite

`benchmark_samples/spotter_benchmark.json` benchmarks what Spotter's coach
actually does in production:

- **Real tools.** Every sample offers exactly the tools Spotter's
  deterministic domain router (general / nutrition / compound) exposes for
  that message, with JSON Schema generated from the Zod validators
  production parses arguments with, and schema validation on.
- **Real prompts.** The system prompt is `buildSystemPrompt` output for
  that route — router mode for `spotter_router`, native mode for `native` —
  built for a fixture profile (Alex, lb, confirm-before-logging on).
- **Real traffic.** Every prompt is checked against production's router and
  its three no-model fast paths (macro goals, workout shorthand, today's
  planned workout), so no sample benchmarks a message the model never sees.
- **Grounded episodes.** Up to production's 8 steps against one fixture
  user whose database ids are fixed UUIDs: an edit or delete only scores
  when it targets the exact id a read returned.

| Category | Count | Tests |
|---|---:|---|
| `workout_logging` | 9 | Typos, relative dates, kg, set types, treadmill / StairMaster / bike rules, today's plan |
| `history_reads` | 5 | Reading history, weight, progress, progression, macros before answering |
| `id_grounded_edits` | 7 | Edits and deletes by the exact workout / exercise / set id; refusing to delete what doesn't exist |
| `charts` | 2 | Plotting exactly the fetched points |
| `nutrition` | 8 | Estimates, exact grams, multi-meal single calls, branded + barcode lookups, saved and plan meals |
| `body_metrics` | 2 | Weigh-ins (flat or batch) and dated backfills |
| `device_actions` | 5 | Rest timer start / default / cancel, form and off-topic videos |
| `plans` | 4 | PPL in one turn, editing the saved plan by id, one-off sessions, showing today's plan |
| `settings` | 3 | Questions vs changes, only the changed field |
| `clarify_or_decline` | 6 | Bare loads, advice, vague logs, chat, the calorie safety floor |
| `memory_and_privacy` | 2 | Remembering an injury; never storing a medication |
| `compound` | 2 | Two deliverables in one message |
| `conversation` | 15 | 3–5 turn dialogues scored on the final world (sets, deletes, memories) |

Refresh after Spotter changes (tools, schemas, prompts, routing):

```bash
npm --prefix spotter install
node spotter/sync_from_gym_bro.mjs ../gym-bro           # export the snapshot
uv run benchmark generate-samples --suite spotter        # rebuild samples
uv run benchmark spotter-routes /tmp/routes.json
node spotter/sync_from_gym_bro.mjs ../gym-bro --check-routes /tmp/routes.json
uv run benchmark run examples/config.spotter.example.json
```

The export fails loudly when a tool is added, renamed, or removed in
Spotter; see [`spotter/README.md`](spotter/README.md).

### Conversation track

`conversation` samples are dialogues of three to five user turns on the
same fixture user. A run passes when the **final world** matches
`expected_state`: which sets exist and at what weight, which fixture ids
were deleted, which memories were stored, and which profile fields
changed. The calls are written on the run and shown in the failure
appendix, and they are not compared to a gold trace. Reading with
`search_workouts` or `get_recent_workouts`, or logging two lifts as one
session or two, both pass when they leave the same state.

Each user turn gets its own `max_steps` budget (production's 8). A
terminal proposal commits at the end of that turn, so the next message
can look the write up. A same-day duplicate of an exercise is rejected
with the existing set ids, and the model has to edit instead. Reads
answer from the committed world, not from the static script. Schema-invalid
calls are returned as an error and do not commit, so the model can retry
inside the turn.

`benchmark spotter-routes` writes every turn, not only the first message,
for the production route check.

## The generic dataset

Exactly **67 samples**, each a standalone JSON file under
`benchmark_samples/tool_calling/`, combined into
`benchmark_samples/full_benchmark.json`. Categories:

| Category | Count | Tests |
|---|---:|---|
| `single_tool` | 20 | One required call with precise arguments |
| `multi_tool` | 10 | Several different tools in one turn (incl. `0067_multi_tool_complex_workflow`) |
| `parallel_tools` | 8 | Independent same-tool fan-out calls |
| `no_tool_decoy` | 8 | Zero calls expected; any call fails the run |
| `argument_precision` | 10 | Exact dates, units, IDs, recipients, enums, nested objects, array order |
| `ordering_required` | 6 | Calls must appear in a required order |
| `tool_choice_under_ambiguity` | 5 | Similar/competing tools; the right one must be chosen |

Tools covered: weather (current + forecast), calendar search/create/update,
email search/draft/send, contacts, web search, calculator, currency and
unit conversion, current time, file search, document retrieval, spreadsheet
update, GitHub issues and PRs, Slack channel + DM mocks, reminders,
geocoding, restaurants, flight and hotel search mocks, code execution mock,
database query mock, CRM update mock, refund lookup mock, and support
tickets. All are **mock schemas**; the benchmark judges the call objects a
model emits and leaves execution out entirely.

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
| `reliability_bar` | `0.95` | Success rate a model needs on a category to be that workload's pick |
| `models` | — | List of model/provider entries |

Each model entry also accepts:

| Field | Meaning |
|---|---|
| `label` | Display name; also keeps two entries for one model (say, two temperatures) separate |
| `protocol` | `native` (default) or `spotter_router` |
| `endpoint_env` | Read the endpoint from this env var instead of `endpoint` |
| `pricing` | `{"input_per_mtok", "output_per_mtok", "cached_input_per_mtok"?}` in USD per million tokens |

Per-model `api_params` are merged into every request for that model
(overriding computed values); a JSON `null` value **removes** a parameter —
useful for models that reject `temperature`, or reasoning deployments that
reject the router's `response_format`.

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

**Azure AI Foundry** (Spotter's deployment) — its `/openai/v1/` base URL is
OpenAI-compatible; see `examples/config.spotter.example.json`:

```json
{ "label": "grok-4-1-fast · router", "name": "grok-4-1-fast-non-reasoning",
  "provider": "openai_compatible", "protocol": "spotter_router",
  "api_key_env": "AZURE_FOUNDRY_API_KEY", "endpoint_env": "AZURE_FOUNDRY_ENDPOINT",
  "pricing": { "input_per_mtok": 0.20, "cached_input_per_mtok": 0.05, "output_per_mtok": 0.50 } }
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
config or sample set always starts fresh. The runner also refuses to
silently overwrite completed runs: with `--no-resume` and an existing
checkpoint it aborts with instructions, and `--force` explicitly discards
the checkpoint and reruns everything.

```bash
uv run benchmark run examples/config.example.json --resume     # default
uv run benchmark run examples/config.example.json --no-resume  # refuse to reuse
uv run benchmark run examples/config.example.json --force      # rerun all
```

## Reproducibility

Everything the benchmark controls is deterministic:

- model, sample, and run ordering are fixed (config order → sample id →
  run index);
- scoring is pure data comparison;
- default temperature is `0`;
- the exact redacted config is snapshotted to
  `outputs/run_config.snapshot.json` together with the config hash, sample
  hash, git commit (when available), Python version, package version,
  OS/platform, seed, and UTC start/finish times;
- results embed the same metadata in `summary`.

**Caveat:** provider APIs may still produce nondeterministic outputs even
at temperature 0 (backend changes, sampling implementation details,
hardware). The benchmark inputs, ordering, scoring, config snapshot, sample
hash, and output formats are fully reproducible, while individual model
outputs can vary between executions.

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
(a single calculation path keeps the report and the JSON in lockstep). It
contains generation timestamps, git commit, version, config path + hash,
sample hash, run counts, an overall ranking sorted by success rate,
a cost & latency table (tokens per run including cached input, cost per
run and per successful run, p50/p95 latency, steps, router recoveries),
workload picks (per category, the cheapest model at or above
`reliability_bar`), per-model / per-category / per-sample tables, a failure
breakdown, the most commonly missed tools, the most common argument and
schema errors, the redacted model configuration summary, a reproducibility
section, and an appendix with every failed run (expected calls, actual
calls, a step trace for episodes, raw output when saved, text output, and
the scorer's reason).

## Tests

```bash
uv run pytest
```

The test suite runs fully offline: provider normalization is exercised
against recorded response shapes, and the checkpoint/runner tests use a
fake in-process provider.

## Project layout

```text
benchmark_samples/   generic: tool_calling/ + full_benchmark.json
                     spotter: spotter/ + spotter_benchmark.json
examples/            example run configs (incl. config.spotter.example.json)
prompts/             default system prompt (generic suite)
spotter/             gym-bro exporter + the exported snapshot
analysis/            output schema docs + example report
src/benchmark/       package: cli, config, schema, runner, episode, router,
                     scoring, world, checkpoint, reporting, samples,
                     spotter_samples, tool_schemas, providers/, utils/
tests/               unit tests (offline; scripted fake providers)
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
- Gemini/Vertex function declarations reject `additionalProperties`; the
  adapter strips it when converting schemas (samples still validate the
  strict schema shape).
- Vertex AI authenticates through Application Default Credentials; set
  `VERTEXAI_PROJECT_ID` and `VERTEXAI_LOCATION` before running.
- OpenRouter normalizes many vendors to the OpenAI shape, but vendor quirks
  upstream of OpenRouter can still surface as provider errors — these are
  tracked separately as `provider_error_failures`.
- Spotter suite scores the model's own decisions. Production adds
  heuristic guards on top (claim, compound, bare-load, delete, and table
  checks) and feeds Zod errors back for a retry; neither is replayed on
  call-scored samples, so those scores are a first-pass floor.
  Conversation samples do return a schema error inside the turn and score
  the world after it. Router-protocol name canonicalization and
  invalid-JSON / unknown-tool nudges *are* replayed.
- Conversation samples commit each terminal proposal at the end of the
  user turn. They do not model the confirm-card tap as its own turn, and
  they do not undo a write the model already committed.
- JSON Schema can't express Zod refinements ("at least one field",
  "targetId unless target is all"), so schema validation catches type,
  range, enum, format, and extra-key errors but not cross-field rules.
- Production canonicalizes router tool names against its full catalog and
  answers an off-route tool with an error result; the benchmark treats an
  off-route name as unknown (same one-time nudge).
