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
        {s.category.value for s in samples}
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
