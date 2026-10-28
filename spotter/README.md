# Spotter snapshot

The Spotter suite benchmarks Spotter's production agent contract. This
folder holds the exporter that reads it straight from a
[gym-bro](https://github.com/Hakeem-Hannoon/gym-bro) checkout, and the
snapshot it last produced.

```text
sync_from_gym_bro.mjs   the exporter (Node; esbuild + zod + zod-to-json-schema)
snapshot/tools.json     42 tools: name, write and terminal flags, catalog
                        description, JSON Schema of the Zod validator, routes
snapshot/prompts/       buildSystemPrompt() for general / nutrition /
                        compound × router (production) / native tool mode
```

## Refresh

```bash
npm --prefix spotter install                    # once
node spotter/sync_from_gym_bro.mjs ../gym-bro   # rewrite snapshot/
uv run benchmark generate-samples --suite spotter
```

gym-bro does not need `npm install`: the exporter bundles
`apps/web/lib/agent/{schemas,tools,prompts,domain-route}.ts` with esbuild,
resolves `@/` through `apps/web/tsconfig.json` and `@spotter/shared` to
`packages/shared/src`, and stubs every other package. The prompts are built
for a fixed date (`--today`, default `2026-10-21`) and the fixture profile in
the script, which mirrors the throwaway user in
`apps/web/tests/live-agent-eval.test.ts`.

Rebuilding the samples after a snapshot change changes the sample hash, so
the next benchmark run starts a fresh checkpoint.

## When the export fails

The script stops when Spotter's `TOOL_CATALOG` and its two maps disagree:

- `TOOL_SCHEMAS` — which Zod export `dispatchTool` validates each tool's
  arguments with (an array means the tool accepts several shapes).
- `DATA_TOOLS` — read tools whose result goes back to the model; every other
  tool ends the router turn.

Add, rename, or remove the entry, re-run, and add or adjust samples in
`src/benchmark/spotter_samples.py` for the new behaviour.

## Checking samples against production

```bash
uv run benchmark spotter-routes /tmp/routes.json
node spotter/sync_from_gym_bro.mjs ../gym-bro --check-routes /tmp/routes.json
```

This runs production's `resolveAgentDomain` on every sample prompt and fails
when a sample declares a different route, or when one of the no-model fast
paths (macro goals, workout shorthand, today's planned workout) would answer
it — a sample like that would benchmark traffic the model never sees.
Conversation samples are checked per turn (`<id>#2`, `<id>#3`, …).
