"""Resumable checkpoint store.

Every completed model/sample/run combination is persisted immediately:

* ``outputs/checkpoints/records.jsonl`` — one JSON line per completed run
  (append-only, crash-safe; a partial trailing line is ignored on load).
* ``outputs/checkpoints/checkpoint.json`` — the main checkpoint file with
  run identity (benchmark version, config hash, sample hash) and the set of
  completed run keys.

A checkpoint key uniquely identifies benchmark version + config hash +
sample hash + provider + model + sample id + run index, so re-running with
a changed config or changed samples never reuses stale results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import RunRecord
from .utils.json_utils import atomic_write_json
from .utils.time import utc_now_iso

CHECKPOINT_FILENAME = "checkpoint.json"
RECORDS_FILENAME = "records.jsonl"


@dataclass(frozen=True)
class RunIdentity:
    """Everything that must match for a checkpointed run to be reusable."""

    benchmark_version: str
    config_hash: str
    sample_hash: str

    def key_prefix(self) -> str:
        return "|".join((self.benchmark_version, self.config_hash,
                         self.sample_hash))


def make_run_key(identity: RunIdentity, provider: str, model_key: str,
                 sample_id: str, run_index: int) -> str:
    return "|".join((identity.key_prefix(), provider, model_key, sample_id,
                     str(run_index)))


class CheckpointConflictError(RuntimeError):
    """Completed runs exist but neither --resume nor --force was given."""


class CheckpointStore:
    def __init__(self, checkpoint_dir: str | Path, identity: RunIdentity):
        self.dir = Path(checkpoint_dir)
        self.identity = identity
        self.checkpoint_path = self.dir / CHECKPOINT_FILENAME
        self.records_path = self.dir / RECORDS_FILENAME
        self._records: dict[str, RunRecord] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, checkpoint_dir: str | Path, identity: RunIdentity, *,
             resume: bool, force: bool) -> "CheckpointStore":
        store = cls(checkpoint_dir, identity)
        store.dir.mkdir(parents=True, exist_ok=True)
        if force:
            store._clear()
            store._write_index()
            return store
        existing = store._load_matching_records()
        if existing and not resume:
            raise CheckpointConflictError(
                f"{len(existing)} completed run(s) exist in "
                f"{store.records_path} for this exact config and sample set. "
                f"Pass --resume to skip them or --force to discard and rerun "
                f"everything."
            )
        store._records = existing
        store._write_index()
        return store

    def _clear(self) -> None:
        self._records = {}
        if self.records_path.exists():
            self.records_path.unlink()

    def _load_matching_records(self) -> dict[str, RunRecord]:
        """Read records.jsonl, keeping only entries for the current identity.

        Entries from older configs/sample sets stay in the file but are
        ignored, so switching configs never corrupts or reuses stale runs.
        """
        records: dict[str, RunRecord] = {}
        if not self.records_path.exists():
            return records
        prefix = self.identity.key_prefix() + "|"
        with open(self.records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry["key"]
                    if not key.startswith(prefix):
                        continue
                    records[key] = RunRecord.model_validate(entry["record"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    # A torn write from an interrupted run — skip the line;
                    # that run simply reruns.
                    continue
        return records

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def key_for(self, provider: str, model_key: str, sample_id: str,
                run_index: int) -> str:
        return make_run_key(self.identity, provider, model_key, sample_id,
                            run_index)

    def has(self, key: str) -> bool:
        return key in self._records

    def get(self, key: str) -> RunRecord | None:
        return self._records.get(key)

    def add(self, key: str, record: RunRecord) -> None:
        """Persist one completed run: append the record, refresh the index."""
        self._records[key] = record
        entry = {"key": key, "record": record.model_dump(mode="json")}
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False,
                               separators=(",", ":")))
            f.write("\n")
            f.flush()
        self._write_index()

    def completed(self) -> dict[str, RunRecord]:
        return dict(self._records)

    def _write_index(self) -> None:
        atomic_write_json(self.checkpoint_path, {
            "benchmark_version": self.identity.benchmark_version,
            "config_hash": self.identity.config_hash,
            "sample_hash": self.identity.sample_hash,
            "updated_utc": utc_now_iso(),
            "records_file": RECORDS_FILENAME,
            "completed_run_count": len(self._records),
            "completed_run_keys": sorted(self._records.keys()),
        })
