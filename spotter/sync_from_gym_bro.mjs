#!/usr/bin/env node
/**
 * Export Spotter's live agent contract into spotter/snapshot/ so the Spotter
 * suite benchmarks exactly what production sends the model.
 *
 *   npm --prefix spotter install
 *   node spotter/sync_from_gym_bro.mjs ../gym-bro [--today 2026-10-21]
 *   node spotter/sync_from_gym_bro.mjs ../gym-bro --check-routes routes.json
 *
 * What it writes:
 *   snapshot/tools.json          every TOOL_CATALOG entry: name, write flag,
 *                                terminal flag, catalog description, the JSON
 *                                Schema of the Zod schema dispatchTool parses
 *                                its arguments with, and the per-domain tool
 *                                lists from domain-route.ts
 *   snapshot/prompts/*.txt       buildSystemPrompt() output for each domain
 *                                (general / nutrition / compound) in both tool
 *                                modes (router = production, native = the
 *                                dormant function-calling path)
 *
 * How: esbuild bundles schemas.ts, tools.ts, prompts.ts and domain-route.ts
 * straight from the gym-bro checkout. `@/` resolves through apps/web's
 * tsconfig paths, `@spotter/shared` to packages/shared/src, zod and
 * zod-to-json-schema to this folder's node_modules, and every other package
 * (supabase, next, server-only, ...) to an inert stub — the exported values
 * are built at module load from plain zod and strings, so nothing runtime is
 * needed. gym-bro does not need `npm install` for this to work.
 *
 * The script fails loudly when the catalog and the maps below drift apart
 * (a tool added, renamed or removed in Spotter), so a stale snapshot can't
 * slip through silently.
 *
 * --check-routes FILE  runs production's resolveAgentDomain on every sample
 *                      prompt in FILE ({id: {prompt, domain}}, written by
 *                      `benchmark spotter-routes`) and exits non-zero when a
 *                      sample's declared route differs, or when one of the
 *                      deterministic fast paths (macro goals, workout
 *                      shorthand, today's planned workout) would answer the
 *                      prompt without a model call — such a sample would
 *                      benchmark traffic the model never sees. Writes nothing.
 */

import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// ---------------------------------------------------------------- arguments

const argv = process.argv.slice(2);
let repoArg = null;
let today = "2026-10-21"; // fixed "today" so dated samples stay deterministic
let checkRoutes = null;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--today") today = argv[++i];
  else if (argv[i] === "--check-routes") checkRoutes = argv[++i];
  else if (!repoArg) repoArg = argv[i];
}
if (!repoArg) {
  console.error("usage: node spotter/sync_from_gym_bro.mjs <path-to-gym-bro> [--today YYYY-MM-DD]");
  process.exit(2);
}
if (!/^\d{4}-\d{2}-\d{2}$/.test(today)) {
  console.error(`--today must be YYYY-MM-DD, got ${today}`);
  process.exit(2);
}
const REPO = path.resolve(repoArg);
const WEB = path.join(REPO, "apps", "web");
const AGENT = path.join(WEB, "lib", "agent");
const SHARED_SRC = path.join(REPO, "packages", "shared", "src");
for (const p of [AGENT, SHARED_SRC]) {
  if (!existsSync(p)) {
    console.error(`not a gym-bro checkout: missing ${p}`);
    process.exit(2);
  }
}

// ------------------------------------------------- Spotter-side knowledge
//
// Which Zod schema dispatchTool (lib/agent/tools.ts) validates each tool's
// arguments with. Tools that parse leniently or take no args map to their
// canonical *ArgsSchema.
const TOOL_SCHEMAS = {
  search_workouts: "searchWorkoutsArgsSchema",
  get_recent_workouts: "getRecentWorkoutsArgsSchema",
  get_exercise_history: "getExerciseHistoryArgsSchema",
  get_progress_summary: "getProgressSummaryArgsSchema",
  get_bodyweight_history: "getBodyweightHistoryArgsSchema",
  plot_workout_data: "plotArgsSchema",
  identify_equipment_from_image: "identifyEquipmentArgsSchema",
  find_exercise_videos: "findExerciseVideosSchema",
  control_rest_timer: "controlRestTimerSchema",
  remember: "rememberSchema",
  forget: "forgetSchema",
  get_workout_plan: "getWorkoutPlanArgsSchema",
  get_progression_suggestions: "getWorkoutPlanArgsSchema", // no args
  get_adaptive_targets: "getAdaptiveTargetsArgsSchema",
  show_saved_plan: "showSavedPlanSchema",
  show_workout_plan: "sessionPlanSchema",
  show_table: "showTableSchema",
  create_pending_workout_session: "parsedWorkoutSchema",
  log_planned_workout: "getWorkoutPlanArgsSchema", // no args
  create_pending_workout_edit: "editWorkoutSchema",
  create_pending_workout_delete: "deleteWorkoutSchema",
  create_pending_exercise_rename: "renameExerciseSchema",
  create_exercise_alias: "createAliasSchema",
  get_user_profile: "getUserProfileArgsSchema",
  update_user_profile: "updateProfileSchema",
  // Accepts one reading or { entries: [...] }; the batch is the canonical shape.
  create_pending_body_metric: "bodyMetricBatchSchema",
  create_pending_plan: "agentPlanSchema",
  set_active_plan: "setActivePlanArgsSchema",
  create_pending_macro_log: "parsedMacroSchema",
  lookup_nutrition: "lookupNutritionArgsSchema",
  lookup_barcode: "lookupBarcodeArgsSchema",
  list_saved_meals: "listSavedMealsArgsSchema",
  save_named_meal: "saveNamedMealArgsSchema",
  log_saved_meal: "logSavedMealArgsSchema",
  log_plan_meal: "logPlanMealArgsSchema",
  delete_saved_meal: "deleteSavedMealArgsSchema",
  get_macros_today: "getMacrosArgsSchema",
  set_macro_goals: "setMacroGoalsSchema",
  show_macro_summary: "getMacrosArgsSchema",
  get_nutrition_profile: "getNutritionProfileArgsSchema",
  get_meal_plans: "getMealPlansArgsSchema",
  create_pending_meal_plan: "agentMealPlanSchema",
};

// Read tools whose result is fed back to the model (ToolOutcome "model_data").
// Every other tool ends the router turn: writes record a pending card, and
// display tools (plot, videos, timer, show_*) render a final response.
const DATA_TOOLS = new Set([
  "search_workouts",
  "get_recent_workouts",
  "get_exercise_history",
  "get_progress_summary",
  "get_bodyweight_history",
  "get_macros_today",
  "lookup_nutrition",
  "lookup_barcode",
  "list_saved_meals",
  "get_nutrition_profile",
  "get_meal_plans",
  "get_adaptive_targets",
  "get_progression_suggestions",
  "get_workout_plan",
  "get_user_profile",
  "remember",
  "forget",
]);

// The profile every exported prompt is built for. Mirrors the throwaway user
// in apps/web/tests/live-agent-eval.test.ts so the two evals agree.
const FIXTURE_PROFILE = {
  displayName: "Alex",
  preferredWeightUnit: "lb",
  trainingGoal: "get stronger",
  equipmentAccess: "full gym",
  confirmBeforeLogging: true,
  humourLevel: 5,
  responseLength: "normal",
};
const MODE = "gym_bro";
const DOMAINS = ["general", "nutrition", "compound"];
const TOOL_MODES = { spotter_router: "router", native: "native" };

// ------------------------------------------------------------------ bundle

const STUB_NS = "spotter-stub";
const STUB_SOURCE = `
const handler = {
  get(_t, prop) {
    if (prop === "__esModule") return true;
    if (prop === Symbol.toPrimitive) return () => "";
    if (prop === "then") return undefined; // never look like a promise
    return stub;
  },
  apply() { return stub; },
  construct() { return stub; },
};
const stub = new Proxy(function () {}, handler);
module.exports = stub;
`;

const resolvePlugin = {
  name: "spotter-resolve",
  setup(b) {
    b.onResolve({ filter: /^(zod|zod-to-json-schema)(\/.*)?$/ }, (args) => ({
      path: require.resolve(args.path),
    }));
    b.onResolve({ filter: /^@spotter\/shared(\/.*)?$/ }, (args) => {
      const sub = args.path.slice("@spotter/shared".length).replace(/^\//, "");
      return { path: path.join(SHARED_SRC, sub ? `${sub}.ts` : "index.ts") };
    });
    // Leave relative paths and the `@/` alias to esbuild (tsconfig paths).
    b.onResolve({ filter: /^[^./]/ }, (args) => {
      if (args.path.startsWith("@/")) return undefined;
      return { path: args.path, namespace: STUB_NS };
    });
    b.onLoad({ filter: /.*/, namespace: STUB_NS }, () => ({
      contents: STUB_SOURCE,
      loader: "js",
    }));
  },
};

const entry = [
  `export * as schemas from ${JSON.stringify(path.join(AGENT, "schemas.ts"))};`,
  `export { TOOL_CATALOG } from ${JSON.stringify(path.join(AGENT, "tools.ts"))};`,
  `export { buildSystemPrompt } from ${JSON.stringify(path.join(AGENT, "prompts.ts"))};`,
  `export { toolNamesForDomain, resolveAgentDomain } from ${JSON.stringify(path.join(AGENT, "domain-route.ts"))};`,
  `export { parseMacroGoalsText } from ${JSON.stringify(path.join(AGENT, "macro-goals-text-parser.ts"))};`,
  `export { parseWorkoutShorthand, FAST_PATH_CONFIDENCE } from ${JSON.stringify(path.join(AGENT, "workout-text-parser.ts"))};`,
  `export { didTodaysPlan } from ${JSON.stringify(path.join(AGENT, "planned-workout.ts"))};`,
  `export { zodToJsonSchema } from "zod-to-json-schema";`,
].join("\n");

const tmp = mkdtempSync(path.join(os.tmpdir(), "spotter-sync-"));
const outfile = path.join(tmp, "bundle.cjs");
let mod;
try {
  await build({
    stdin: { contents: entry, resolveDir: WEB, loader: "ts", sourcefile: "spotter-entry.ts" },
    bundle: true,
    platform: "node",
    format: "cjs",
    target: "node18",
    outfile,
    tsconfig: path.join(WEB, "tsconfig.json"),
    plugins: [resolvePlugin],
    logLevel: "error",
  });
  mod = require(outfile);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

// ----------------------------------------------------------- route check

if (checkRoutes) {
  const samples = JSON.parse(readFileSync(checkRoutes, "utf8"));
  const gate = mod.FAST_PATH_CONFIDENCE;
  const wrong = [];
  const intercepted = [];
  for (const [id, { prompt, domain }] of Object.entries(samples)) {
    const actual = mod.resolveAgentDomain({ message: prompt });
    if (actual !== domain) wrong.push(`  ${id}: declared ${domain}, production routes ${actual} — "${prompt}"`);
    const goals = mod.parseMacroGoalsText(prompt);
    const shorthand = mod.parseWorkoutShorthand(prompt);
    const path_ =
      goals && goals.confidence >= gate ? "macro-goals"
      : shorthand && shorthand.confidence >= gate ? "workout shorthand"
      : mod.didTodaysPlan(prompt) ? "planned workout"
      : null;
    if (path_) intercepted.push(`  ${id}: the ${path_} fast path answers this without the model — "${prompt}"`);
  }
  if (wrong.length) console.error(`${wrong.length} sample(s) declare the wrong route:\n${wrong.join("\n")}`);
  if (intercepted.length) console.error(`${intercepted.length} sample(s) never reach the model:\n${intercepted.join("\n")}`);
  if (wrong.length || intercepted.length) process.exit(1);
  console.log(`all ${Object.keys(samples).length} sample routes match production and reach the model`);
  process.exit(0);
}

// ------------------------------------------------------------ validation

const catalog = mod.TOOL_CATALOG;
if (!Array.isArray(catalog) || catalog.length === 0) {
  throw new Error("TOOL_CATALOG did not export as a non-empty array");
}
const catalogNames = catalog.map((t) => t.name);
const unmapped = catalogNames.filter((n) => !(n in TOOL_SCHEMAS));
const stale = Object.keys(TOOL_SCHEMAS).filter((n) => !catalogNames.includes(n));
const staleData = [...DATA_TOOLS].filter((n) => !catalogNames.includes(n));
if (unmapped.length || stale.length || staleData.length) {
  throw new Error(
    "Spotter's tool catalog drifted from this script. Update TOOL_SCHEMAS / DATA_TOOLS:\n" +
      (unmapped.length ? `  new tools with no schema mapping: ${unmapped.join(", ")}\n` : "") +
      (stale.length ? `  mapped tools no longer in the catalog: ${stale.join(", ")}\n` : "") +
      (staleData.length ? `  DATA_TOOLS entries no longer in the catalog: ${staleData.join(", ")}\n` : ""),
  );
}

const domainTools = Object.fromEntries(
  DOMAINS.map((d) => [d, [...mod.toolNamesForDomain(d)]]),
);

function jsonSchemaFor(toolName) {
  const exportName = TOOL_SCHEMAS[toolName];
  const zodSchema = mod.schemas[exportName];
  if (!zodSchema || typeof zodSchema.safeParse !== "function") {
    throw new Error(`schemas.ts has no zod export ${exportName} (for ${toolName})`);
  }
  const schema = mod.zodToJsonSchema(zodSchema, {
    $refStrategy: "none",
    target: "jsonSchema7",
  });
  delete schema.$schema;
  if (schema.type !== "object") {
    throw new Error(`${toolName}: ${exportName} is not an object schema`);
  }
  schema.properties ??= {};
  return schema;
}

const tools = catalog.map((t) => ({
  name: t.name,
  write: Boolean(t.write),
  terminal: !DATA_TOOLS.has(t.name),
  description: t.description,
  zod_schema: TOOL_SCHEMAS[t.name],
  domains: DOMAINS.filter((d) => domainTools[d].includes(t.name)),
  parameters: jsonSchemaFor(t.name),
}));

// ---------------------------------------------------------------- prompts

const prompts = {};
for (const domain of DOMAINS) {
  for (const [protocol, toolMode] of Object.entries(TOOL_MODES)) {
    const text = mod.buildSystemPrompt({
      profile: FIXTURE_PROFILE,
      todayIso: today,
      mode: MODE,
      memories: [],
      toolMode,
      domain,
      tables: true,
    });
    if (typeof text !== "string" || !text.includes("Available tools:")) {
      throw new Error(`buildSystemPrompt(${domain}, ${toolMode}) returned no tool catalog`);
    }
    prompts[`${domain}.${protocol}`] = text;
  }
}

// ----------------------------------------------------------------- write

function gitHead(dir) {
  try {
    return execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

const outDir = path.join(HERE, "snapshot");
const promptDir = path.join(outDir, "prompts");
rmSync(promptDir, { recursive: true, force: true });
mkdirSync(promptDir, { recursive: true });

const snapshot = {
  source: "Hakeem-Hannoon/gym-bro",
  gym_bro_commit: gitHead(REPO),
  today,
  coach_mode: MODE,
  fixture_profile: FIXTURE_PROFILE,
  tool_count: tools.length,
  domains: domainTools,
  prompts: Object.fromEntries(
    Object.keys(prompts).map((id) => [id, `prompts/${id}.txt`]),
  ),
  tools,
};
writeFileSync(path.join(outDir, "tools.json"), JSON.stringify(snapshot, null, 2) + "\n");
for (const [id, text] of Object.entries(prompts)) {
  writeFileSync(path.join(promptDir, `${id}.txt`), text.endsWith("\n") ? text : text + "\n");
}

console.log(
  `exported ${tools.length} tools and ${Object.keys(prompts).length} prompts ` +
    `from gym-bro@${(snapshot.gym_bro_commit ?? "unknown").slice(0, 7)} → ${path.relative(process.cwd(), outDir)}`,
);
