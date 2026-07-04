"""Sample set integrity and validator behavior."""

import copy

import pytest

from benchmark.samples import (
    EXPECTED_SAMPLE_COUNT, SampleValidationError, build_all_samples,
    generate_sample_files, load_benchmark_file, samples_hash,
    validate_samples,
)
from benchmark.schema import Category

REQUIRED_CATEGORIES = {c.value for c in Category}


def test_exactly_67_samples():
    assert len(build_all_samples()) == EXPECTED_SAMPLE_COUNT == 67


def test_all_samples_validate():
    samples = validate_samples(build_all_samples())
    assert len(samples) == 67


def test_ids_unique_and_sorted_format():
    ids = [s["id"] for s in build_all_samples()]
    assert len(set(ids)) == 67
    assert ids == sorted(ids)
    assert ids[0] == "0001_weather_current"
    assert ids[-1] == "0067_multi_tool_complex_workflow"


def test_every_category_present():
    categories = {s["category"] for s in build_all_samples()}
    assert categories == REQUIRED_CATEGORIES


def test_every_difficulty_present():
    difficulties = {s["difficulty"] for s in build_all_samples()}
    assert difficulties == {"easy", "medium", "hard"}


def test_no_tool_decoys_expect_zero_calls():
    for sample in build_all_samples():
        if sample["category"] == "no_tool_decoy":
            assert sample["expected_tool_calls"] == []
            assert sample["tools"], "decoys must still offer tools"


def test_expected_calls_reference_offered_tools():
    for sample in validate_samples(build_all_samples()):
        offered = {t.function.name for t in sample.tools}
        for expected in sample.expected_tool_calls:
            assert expected.name in offered


def test_validator_rejects_unknown_category():
    bad = copy.deepcopy(build_all_samples())
    bad[0]["category"] = "not_a_category"
    with pytest.raises(SampleValidationError):
        validate_samples(bad)


def test_validator_rejects_missing_prompt():
    bad = copy.deepcopy(build_all_samples())
    del bad[3]["prompt"]
    with pytest.raises(SampleValidationError):
        validate_samples(bad)


def test_validator_rejects_expected_call_to_unoffered_tool():
    bad = copy.deepcopy(build_all_samples())
    bad[0]["expected_tool_calls"][0]["name"] = "no_such_tool"
    with pytest.raises(SampleValidationError):
        validate_samples(bad)


def test_validator_rejects_duplicate_ids():
    bad = copy.deepcopy(build_all_samples())
    bad[1]["id"] = bad[0]["id"]
    with pytest.raises(SampleValidationError):
        validate_samples(bad)


def test_validator_rejects_malformed_scoring_flag():
    bad = copy.deepcopy(build_all_samples())
    bad[0]["scoring"]["require_order"] = "yes"  # not a bool
    with pytest.raises(SampleValidationError):
        validate_samples(bad)


def test_generate_and_load_roundtrip(tmp_path):
    combined = generate_sample_files(tmp_path / "benchmark_samples")
    assert combined.exists()
    per_file_dir = tmp_path / "benchmark_samples" / "tool_calling"
    assert len(list(per_file_dir.glob("*.json"))) == 67
    loaded = load_benchmark_file(combined)
    assert len(loaded) == 67
    # Hash is stable across generate/load and matches the in-code set.
    in_code = validate_samples(build_all_samples())
    assert samples_hash(loaded) == samples_hash(in_code)


def test_load_supports_path_references(tmp_path):
    base = tmp_path / "benchmark_samples"
    generate_sample_files(base)
    ref_file = base / "refs_benchmark.json"
    refs = [f"tool_calling/{p.name}" for p in
            sorted((base / "tool_calling").glob("*.json"))]
    import json
    ref_file.write_text(json.dumps({"samples": refs}), encoding="utf-8")
    loaded = load_benchmark_file(ref_file)
    assert len(loaded) == 67
