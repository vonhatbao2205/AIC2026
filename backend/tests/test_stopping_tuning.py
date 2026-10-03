"""DEV replay cannot use test labels or silently bypass evidence guards."""
import hashlib
import json

import pytest

from benchmarks.agent.metrics import measure
from benchmarks.agent.tune_stopping import RecordedBoard, replay, tune, video_components
from backend.app.agent.decision.calibration import Calibration
from backend.app.agent.evidence.compiler import Constraint


def target():
    return {"video_id": "v", "frame_idx": 100, "fps": 25}


def frame():
    return {**target(), "pts_time": 4, "submit_keyframe_id": "v/1", "probability": .7,
            "raw_probability": .7, "constraint_scores": {"query": {"supported": .7, "unknown": .3, "refuted": 0}}}


def q(qid="q", **extra):
    return {"query_id": qid, "query": "red bicycle", "query_type": "T-KIS", "split": "dev", "targets": [target()], **extra}


def test_replay_keeps_raw_evidence_guard_and_null_probability():
    f = frame()
    row = {"query_id": "q", "variant": "E", "metrics": {"r1": 1, "r5": 1, "codex_calls": 1, "claude_calls": 1, "latency_s": 80}}
    point = {"ranking": [f], "constraints": [Constraint("query", "red bicycle")], "agents": ["claude"], "at_s": 30}
    outcome = replay(row, q(), [point], Calibration(5), .5, .1, 1)
    assert outcome["agents"] == 1 and outcome["r1"] == 1
    f["constraint_scores"]["query"].update(supported=.4, unknown=.6)
    assert replay(row, q(), [point], Calibration(.1), .5, 0, 1)["agents"] == 2
    f["probability"] = None
    assert replay(row, q(), [point], Calibration(), .5, 0, 1)["agents"] == 2


def test_crossfit_video_groups_are_transitive():
    queries = {"a": q("a", targets=[{"video_id": "v1"}]),
               "b": q("b", targets=[{"video_id": "v1"}, {"video_id": "v2"}]),
               "c": q("c", targets=[{"video_id": "v2"}]), "d": q("d", targets=[{"video_id": "v3"}])}
    assert video_components(queries)["a"] == {"a", "b", "c"}
    assert video_components(queries)["d"] == {"d"}


def source(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    queries = [q("a"), q("b", query="blue bicycle", targets=[{**target(), "video_id": "other"}]),
               q("c", query="yellow bicycle", targets=[{**target(), "video_id": "third"}])]
    dataset.write_text("\n".join(json.dumps(item) for item in queries))
    run = tmp_path / "main-dev"
    run.mkdir()
    config = {"split": "dev", "mode": "live", "hint_mode": "full", "tolerance_s": 1,
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "code_sha256": "source-code",
              "query_ids": [item["query_id"] for item in queries]}
    (run / "manifest.json").write_text(json.dumps({"config": config, "fingerprint": "manifest"}))
    rows = []
    for query in queries:
        for variant in ["E", "F"]:
            f = frame()
            snapshot = {"ranking": [f], "metrics": {"agent_calls": {"codex": 1, "claude": 1}},
                        "controller": {"status": "done", "calibrated": False, "verifier": "jev", "decision_calls": [{"backend": "jev", "model": "jev"}]}}
            rows.append({"query_id": query["query_id"], "variant": variant, "split": "dev", "hint_stage": "full",
                         "snapshot": snapshot, "metrics": measure(query, snapshot, [], wall_s=80),
                         "trace": {"constraints": [{"id": "query", "text": query["query"]}], "trace": [
                             {"kind": "agent_results", "agents": ["claude"], "at_s": 29},
                             {"kind": "verification", "at_s": 30}, {"kind": "ranking", "at_s": 30, "ranking": [f]}]}})
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return dataset, run, rows


def test_tuner_writes_distinct_aggregation_profiles_and_rejects_test(tmp_path):
    dataset, run, rows = source(tmp_path)
    bundle, report = tune(dataset, run, tmp_path / "fit")
    assert set(bundle["profiles"]) == {"holistic", "constraints"}
    assert bundle["profiles"]["holistic"]["query_ids"] == ["a", "b", "c"]
    assert report["variants"]["E"]["selected_crossfit_replay_healthy"]["queries"] == 3
    assert (tmp_path / "fit/TUNING_REPORT.md").exists()
    rows[0]["split"] = "test"
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="TEST"):
        tune(dataset, run, tmp_path / "reject")


def test_tuner_audits_but_excludes_failed_runs_from_calibration(tmp_path):
    dataset, run, rows = source(tmp_path)
    rows[0]["metrics"]["agent_failed"] = True
    rows[0]["snapshot"]["agents"] = {"claude": {"status": "failed", "error_kind": None,
                                                 "error": "You've hit your session limit"}}
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    bundle, report = tune(dataset, run, tmp_path / "fit")
    assert bundle["profiles"]["holistic"]["query_ids"] == ["b", "c"]
    assert report["variants"]["E"]["excluded_failed_query_ids"] == ["a"]
    assert report["failure_audit"][0]["diagnosed_kind"] == "quota"
