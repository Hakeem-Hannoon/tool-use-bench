"""Results document structure and Markdown report generation."""

import json

from benchmark.config import BenchmarkConfig
from benchmark.reporting import (
    build_results_document, generate_markdown_report, split_summary_files,
)
from benchmark.runner import _record_from_result
from benchmark.samples import build_all_samples, validate_samples
from benchmark.schema import ProviderResult


def make_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate({
        "runs_per_sample": 2,
        "models": [
            {"name": "fake-model", "provider": "openai",
             "api_key_env": "OPENAI_API_KEY"},
        ],
    })


def make_records(config, samples):
    records = []
    for sample in samples:
        for run_index in range(config.runs_per_sample):
            # Second run of the second sample fails (drops all calls).
            fail = sample is samples[1] and run_index == 1
            result = ProviderResult(
                model="fake-model", provider="openai", sample_id=sample.id,
                run_index=run_index,
                raw_response={"stub": True},
                tool_calls=[] if fail else sample.expected_tool_calls,
                text="I will not call tools." if fail else "",
                latency_ms=42,
            )
            records.append(_record_from_result(sample, result, save_raw=True))
    return records


def build_doc():
    config = make_config()
    samples = validate_samples(build_all_samples())[:3]
    records = make_records(config, samples)
    return config, samples, build_results_document(
        config=config, config_path="examples/config.example.json",
        config_hash="sha256:cfg", samples=samples, sample_hash="sha256:smp",
        sample_file="benchmark_samples/full_benchmark.json",
        records=records, started_utc="2026-07-04T00:00:00Z",
        finished_utc="2026-07-04T00:30:00Z",
        generated_utc="2026-07-04T00:30:00Z",
    )


def test_summary_appears_before_runs():
    _, _, doc = build_doc()
    keys = list(doc.keys())
    assert keys[0] == "summary"
    assert keys.index("summary") < keys.index("runs")
    assert doc["summary"]["benchmark_name"] == "tool-call-bench"
    assert doc["summary"]["config_hash"] == "sha256:cfg"
    assert doc["summary"]["sample_hash"] == "sha256:smp"


def test_model_summaries_have_correct_totals():
    config, samples, doc = build_doc()
    (summary,) = doc["model_summaries"]
    expected_total = config.runs_per_sample * len(samples)
    assert summary["total_runs"] == expected_total
    assert summary["successful_runs"] == expected_total - 1
    assert summary["failed_runs"] == 1
    assert summary["missing_tool_call_failures"] == 1
    assert summary["samples_with_at_least_one_failure"] == 1
    assert summary["samples_passed_all_30_runs"] == len(samples) - 1
    rate = summary["successful_runs"] / summary["total_runs"]
    assert abs(summary["success_rate"] - rate) < 1e-9
    assert summary["success_rate_percent"] == f"{rate * 100:.2f}%"


def test_category_and_sample_summaries_cover_all_records():
    config, samples, doc = build_doc()
    assert {c["category"] for c in doc["category_summaries"]} == \
        {s.category for s in samples}
    assert {p["sample_id"] for p in doc["sample_summaries"]} == \
        {s.id for s in samples}
    for entry in doc["sample_summaries"]:
        assert entry["total_runs"] == config.runs_per_sample


def test_failure_statistics_track_missing_tools():
    _, samples, doc = build_doc()
    stats = doc["failure_statistics"]
    assert stats["failure_type_counts"]["missing_tool_call"] == 1
    missing_names = {m["tool_name"] for m in stats["most_common_missing_tools"]}
    assert missing_names == {c.name for c in samples[1].expected_tool_calls}


def test_document_is_json_serializable():
    _, _, doc = build_doc()
    json.dumps(doc)


def test_split_summary_files_only_failed_runs_in_failures():
    _, _, doc = build_doc()
    split = split_summary_files(doc)
    assert set(split) == {"model_summaries.json", "sample_summaries.json",
                          "category_summaries.json", "failures.json"}
    failed = split["failures.json"]["failed_runs"]
    assert len(failed) == 1
    assert all(not run["success"] for run in failed)


def test_markdown_report_normalized_sections():
    _, samples, doc = build_doc()
    md = generate_markdown_report(doc, local_date="2026-07-03T20:00:00-04:00")
    assert md.startswith("# Tool Call Benchmark Report")
    for section in (
        "## Overall Ranking", "## Per-Model Results",
        "## Per-Category Results", "## Per-Sample Results",
        "## Failure Breakdown", "## Most Commonly Missed Tools",
        "## Most Common Argument Errors",
        "## Model Configurations (secrets redacted)",
        "## Reproducibility", "## Appendix: Failed Runs",
    ):
        assert section in md, f"missing section {section}"
    assert "| Rank | Model | Provider | Successful Runs | Total Runs | Success Rate |" in md
    assert "Generated UTC: 2026-07-04T00:30:00Z" in md
    assert "Runs per sample: 2" in md
    # The failing run appears in the appendix with its scorer reason.
    assert samples[1].id in md
    assert "Model produced no tool calls" in md
    # Report is generated from results.json content only — env var names may
    # appear, secret values never exist in the document at all.
    assert "OPENAI_API_KEY" in md


# ---------------------------------------------------------------------------
# Protocol-aware grouping, cost, latency, workload picks
# ---------------------------------------------------------------------------

from benchmark.reporting import percentile, workload_picks  # noqa: E402
from benchmark.schema import Usage  # noqa: E402


def two_protocol_doc():
    config = BenchmarkConfig.model_validate({
        "runs_per_sample": 2,
        "reliability_bar": 0.9,
        "models": [
            {"name": "grok", "provider": "openai_compatible",
             "endpoint_env": "AZURE_FOUNDRY_ENDPOINT", "protocol": "native",
             "pricing": {"input_per_mtok": 0.2, "output_per_mtok": 0.5}},
            {"name": "grok", "provider": "openai_compatible",
             "endpoint_env": "AZURE_FOUNDRY_ENDPOINT",
             "protocol": "spotter_router",
             "pricing": {"input_per_mtok": 0.2, "output_per_mtok": 0.5}},
        ],
    })
    samples = validate_samples(build_all_samples())[:2]
    records = []
    for cfg_index, model_cfg in enumerate(config.models):
        for sample in samples:
            for run_index in range(config.runs_per_sample):
                # The router entry fails every run of the first sample.
                fail = cfg_index == 1 and sample is samples[0]
                result = ProviderResult(
                    model=model_cfg.name, provider=model_cfg.provider,
                    sample_id=sample.id, run_index=run_index,
                    tool_calls=[] if fail else sample.expected_tool_calls,
                    latency_ms=100 * (run_index + 1),
                    usage=Usage(input_tokens=10_000 * (cfg_index + 1),
                                output_tokens=100),
                )
                records.append(_record_from_result(
                    sample, result, save_raw=False, model_cfg=model_cfg))
    doc = build_results_document(
        config=config, config_path="c.json", config_hash="h", samples=samples,
        sample_hash="s", sample_file="f", records=records,
        started_utc="t0", finished_utc="t1", generated_utc="t1")
    return config, samples, doc


def test_same_model_under_two_protocols_stays_separate():
    _, _, doc = two_protocol_doc()
    native, router = doc["model_summaries"]
    assert native["protocol"] == "native" and router["protocol"] == "spotter_router"
    assert native["model_key"] != router["model_key"]
    assert native["success_rate"] == 1.0
    assert router["success_rate"] == 0.5
    assert router["label"] == "grok (spotter_router)"


def test_cost_and_latency_statistics():
    _, samples, doc = two_protocol_doc()
    native, router = doc["model_summaries"]
    # 10k input * 0.2 + 100 output * 0.5 per million = 0.00205 per run
    assert native["cost_per_run_usd"] == 0.00205
    assert native["total_cost_usd"] == 0.00205 * 4
    assert native["cost_per_successful_run_usd"] == 0.00205
    assert router["cost_per_successful_run_usd"] == round(0.00405 * 4 / 2, 8)
    assert native["avg_input_tokens"] == 10_000
    assert native["latency_p50_ms"] == 100 and native["latency_p95_ms"] == 200


def test_workload_picks_cheapest_model_clearing_the_bar():
    _, samples, doc = two_protocol_doc()
    picks = {p["category"]: p for p in doc["workload_picks"]}
    first_category = samples[0].category
    pick = picks[first_category]
    assert pick["clears_bar"] and pick["label"] == "grok"
    rows = [
        {"category": "x", "label": "cheap", "model_key": "a",
         "success_rate": 0.97, "success_rate_percent": "97.00%",
         "cost_per_run_usd": 0.001},
        {"category": "x", "label": "pricey", "model_key": "b",
         "success_rate": 1.0, "success_rate_percent": "100.00%",
         "cost_per_run_usd": 0.01},
        {"category": "y", "label": "flaky", "model_key": "a",
         "success_rate": 0.5, "success_rate_percent": "50.00%",
         "cost_per_run_usd": 0.001},
    ]
    by_cat = {p["category"]: p for p in workload_picks(rows, 0.95)}
    assert by_cat["x"]["label"] == "cheap" and by_cat["x"]["eligible_models"] == 2
    assert by_cat["y"]["clears_bar"] is False


def test_markdown_report_has_cost_and_picks_sections():
    _, _, doc = two_protocol_doc()
    md = generate_markdown_report(doc)
    assert "## Cost & Latency" in md and "## Workload Picks" in md
    assert "grok (spotter_router)" in md
    assert "$AZURE_FOUNDRY_ENDPOINT" in md  # env var name, never a value


def test_percentile_is_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([3, 1, 2, 4], 50) == 2
