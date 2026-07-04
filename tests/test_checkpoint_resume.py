"""Checkpoint persistence, key identity, and end-to-end resume."""

import json

import pytest

import benchmark.runner as runner_module
from benchmark.checkpoint import (
    CheckpointConflictError, CheckpointStore, RunIdentity, make_run_key,
)
from benchmark.config import BenchmarkConfig
from benchmark.providers.base import BaseProvider
from benchmark.runner import run_benchmark
from benchmark.samples import build_all_samples
from benchmark.schema import (
    ProviderResult, RunRecord, ScoringResult, ToolCall,
)

IDENTITY = RunIdentity(benchmark_version="0.1.0", config_hash="sha256:cfg",
                       sample_hash="sha256:smp")


def make_record(sample_id="0001_weather_current", run_index=0) -> RunRecord:
    call = ToolCall(name="get_weather",
                    arguments={"location": "Toronto, Canada",
                               "unit": "celsius"})
    return RunRecord(
        model="fake-model", provider="openai", sample_id=sample_id,
        run_index=run_index, success=True, expected_tool_calls=[call],
        actual_tool_calls=[call],
        scoring_result=ScoringResult(success=True, reason="Passed."),
        latency_ms=1,
    )


def test_checkpoint_key_includes_config_and_sample_hash():
    key = make_run_key(IDENTITY, "openai", "openai/fake-model",
                       "0001_weather_current", 3)
    assert "sha256:cfg" in key
    assert "sha256:smp" in key
    assert "0.1.0" in key
    assert key.endswith("0001_weather_current|3")


def test_add_persists_and_reopen_resumes(tmp_path):
    store = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=False)
    key = store.key_for("openai", "openai/fake-model",
                        "0001_weather_current", 0)
    assert not store.has(key)
    store.add(key, make_record())
    assert store.has(key)
    assert (tmp_path / "checkpoint.json").exists()
    assert (tmp_path / "records.jsonl").exists()

    reopened = CheckpointStore.open(tmp_path, IDENTITY, resume=True,
                                    force=False)
    assert reopened.has(key)
    assert reopened.get(key).success


def test_records_from_other_identity_are_ignored(tmp_path):
    store = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=False)
    key = store.key_for("openai", "openai/fake-model",
                        "0001_weather_current", 0)
    store.add(key, make_record())

    other = RunIdentity(benchmark_version="0.1.0",
                        config_hash="sha256:DIFFERENT",
                        sample_hash="sha256:smp")
    other_store = CheckpointStore.open(tmp_path, other, resume=True,
                                       force=False)
    assert not other_store.completed()


def test_no_resume_with_existing_runs_raises(tmp_path):
    store = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=False)
    store.add(store.key_for("openai", "m", "0001_weather_current", 0),
              make_record())
    with pytest.raises(CheckpointConflictError):
        CheckpointStore.open(tmp_path, IDENTITY, resume=False, force=False)


def test_force_discards_previous_runs(tmp_path):
    store = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=False)
    store.add(store.key_for("openai", "m", "0001_weather_current", 0),
              make_record())
    forced = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=True)
    assert not forced.completed()


def test_torn_trailing_line_is_ignored(tmp_path):
    store = CheckpointStore.open(tmp_path, IDENTITY, resume=True, force=False)
    key = store.key_for("openai", "m", "0001_weather_current", 0)
    store.add(key, make_record())
    with open(tmp_path / "records.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": "half-written')  # simulated crash mid-append
    reopened = CheckpointStore.open(tmp_path, IDENTITY, resume=True,
                                    force=False)
    assert list(reopened.completed()) == [key]


# ---------------------------------------------------------------------------
# End-to-end resume through the runner with a fake provider
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Echoes each sample's expected calls; counts real invocations."""

    name = "fake"
    calls = 0

    def check_credentials(self) -> None:  # no credentials needed
        return

    async def _call(self, prompt, system_prompt, tools):
        raise AssertionError("complete() is overridden")

    async def complete(self, sample, run_index, system_prompt):
        type(self).calls += 1
        return ProviderResult(
            model=self.cfg.name, provider=self.cfg.provider,
            sample_id=sample.id, run_index=run_index,
            raw_response={"fake": True},
            tool_calls=sample.expected_tool_calls, text="", latency_ms=1,
        )


def write_mini_benchmark(tmp_path):
    samples = build_all_samples()[:3]
    path = tmp_path / "mini_benchmark.json"
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return path, samples


def make_run_config(tmp_path, benchmark_path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate({
        "benchmark": str(benchmark_path),
        "output": str(tmp_path / "outputs" / "results.json"),
        "report": str(tmp_path / "reports" / "tool_call_report.md"),
        "runs_per_sample": 2,
        "concurrency": 3,
        "max_retries": 0,
        "timeout_seconds": 10,
        "models": [{"name": "fake-model", "provider": "openai",
                    "api_key_env": None}],
    })


async def test_runner_resume_does_not_rerun_completed_runs(tmp_path,
                                                           monkeypatch):
    benchmark_path, samples = write_mini_benchmark(tmp_path)
    config = make_run_config(tmp_path, benchmark_path)
    monkeypatch.setattr(runner_module, "create_provider",
                        lambda cfg, settings: FakeProvider(cfg, settings))
    FakeProvider.calls = 0

    results1 = await run_benchmark("cfg.json", config, resume=True,
                                   force=False)
    expected_runs = len(samples) * config.runs_per_sample
    assert FakeProvider.calls == expected_runs
    assert results1["summary"]["total_runs_all_models"] == expected_runs
    assert all(run["success"] for run in results1["runs"])

    # Second run with the same config: everything already checkpointed.
    results2 = await run_benchmark("cfg.json", config, resume=True,
                                   force=False)
    assert FakeProvider.calls == expected_runs  # zero new provider calls
    assert results2["summary"]["total_runs_all_models"] == expected_runs

    # --no-resume without --force refuses to touch completed runs.
    with pytest.raises(CheckpointConflictError):
        await run_benchmark("cfg.json", config, resume=False, force=False)

    # --force reruns everything.
    await run_benchmark("cfg.json", config, resume=True, force=True)
    assert FakeProvider.calls == expected_runs * 2

    # All expected output files exist.
    out = tmp_path / "outputs"
    for name in ("results.json", "model_summaries.json",
                 "sample_summaries.json", "category_summaries.json",
                 "failures.json", "run_config.snapshot.json"):
        assert (out / name).exists(), name
    assert (out / "checkpoints" / "checkpoint.json").exists()
    assert (tmp_path / "reports" / "tool_call_report.md").exists()

    snapshot = json.loads((out / "run_config.snapshot.json").read_text())
    assert snapshot["finished_utc"] is not None
    assert snapshot["config"]["models"][0]["api_key_env"] is None
    text = json.dumps(snapshot)
    assert "sk-" not in text  # no secret material anywhere
