import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.agent.metrics import calibration_metrics, first_rank, measure, moment_matches, summarize
from benchmarks.agent.run_benchmark import load_queries, run_one, VARIANTS


def query(**extra):
    return {"query_id": "q", "query": "red bicycle", "query_type": "T-KIS", "split": "test",
            "targets": [{"video_id": "v", "frame_idx": 100, "fps": 25}], **extra}


def f(**extra):
    return {"video_id": "v", "submit_keyframe_id": "v/001", "frame_idx": 100, "pts_time": 4, "fps": 25, **extra}


def test_moment_matching_does_not_accept_right_video_wrong_moment():
    assert first_rank([f(frame_idx=999), f()], query()) == 2
    assert first_rank([f(frame_idx=999)], query()) is None
    assert moment_matches(f(pts_time=4, frame_idx=None), query()["targets"][0], 1)


def test_qa_needs_answer_not_only_retrieval():
    q = query(query_type="QA", targets=[{"video_id": "v", "frame_idx": 100, "answers": ["Đỏ"]}])
    assert first_rank([f()], q) is None
    assert first_rank([f(answer=" đỏ ")], q) == 1
    assert first_rank([f(answer="white")], q) is None


def test_trake_requires_complete_ordered_single_video_alternative():
    q = query(query_type="TRAKE", sequences=[[{"video_id": "v", "frame_idx": 100}, {"video_id": "v", "frame_idx": 200}]])
    a, b = f(event=1), f(event=2, frame_idx=200, pts_time=8)
    assert first_rank([a, b], q) == 1
    assert first_rank([a], q) is None
    assert first_rank([a, {**b, "video_id": "other"}], q) is None
    assert first_rank([a, {**b, "pts_time": 1}], q) is None


def test_metric_rescue_cost_coverage_calibration_and_first_correct():
    snap = {"ranking": [f(probability=.9, agent_sources=["codex"])], "baseline_ranking": [f(frame_idx=900)],
            "agents": {"codex": {"cost_usd": None, "input_tokens": 30, "output_tokens": 20}},
            "metrics": {"agent_calls": {"codex": 1}, "tool_calls": 3}}
    m = measure(query(), snap, [{"kind": "ranking", "at_s": 2, "ranking": [f()]},
                               {"kind": "ranking", "at_s": 8, "ranking": [f()]}], wall_s=10)
    assert m["r1"] == 1 and m["rescue1"]
    assert m["time_to_first_correct_s"] == 2
    assert m["cost_usd"] is None and m["tokens"] == 50
    s = summarize([{"variant": "F", "metrics": m}])["F"]
    assert s["agent_rescue_at_1"] == 1
    assert s["cost_usd_mean"] is None and s["cost_usd_coverage"] == 0
    assert s["jev_calibration"]["brier"] == pytest.approx(.01)


def test_repeated_verification_does_not_duplicate_calibration_observations():
    snap = {"ranking": [f(probability=.5)], "baseline_ranking": [], "metrics": {}}
    m = measure(query(), snap, [{"kind": "verification", "observations": [f()]}] * 5, wall_s=1)
    assert len(m["calibration"]) == 1


def test_empty_calibration_metrics_and_exact_boundaries():
    assert calibration_metrics([])["ece"] is None
    assert calibration_metrics([(0, 0), (1, 1)])["ece"] == 0
    assert calibration_metrics([(0, 1), (1, 0)])["brier"] == 1


def test_dataset_rejects_missing_or_duplicate_labels(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps(query(targets=[])))
    with pytest.raises(ValueError):
        load_queries(p)
    p.write_text(json.dumps(query()) + "\n" + json.dumps(query()))
    with pytest.raises(ValueError):
        load_queries(p)


@pytest.mark.asyncio
async def test_live_runner_never_sends_ground_truth_or_hints():
    import httpx
    bodies = []
    def respond(req):
        if req.method == "POST":
            body = json.loads(req.content)
            bodies.append(body)
            return httpx.Response(200, json={"finished": True, "run_id": "r", "ranking": []})
        return httpx.Response(200, json={"trace": []})
    async with httpx.AsyncClient(base_url="http://local", transport=httpx.MockTransport(respond)) as client:
        for variant in VARIANTS:
            await run_one(client, query(previous_hints=["leaked past query"], answer="leaked answer"), variant, 2, .01)
    assert all("targets" not in b and "answer" not in b and b["previous_hints"] == [] for b in bodies)
    assert {b["policy"] for b in bodies} >= {"retrieval", "codex", "parallel", "rerank", "adaptive", "full"}


@pytest.mark.asyncio
async def test_all_six_variants_through_http_runner(tmp_path, monkeypatch):
    import httpx
    from app import main
    from benchmarks.agent.run_benchmark import write_report
    monkeypatch.setattr(main.agent_service, "_binary", lambda name: "fake")
    async def fake_agent(run, state, binary, url):
        state.begin()
        await main.agent_service.tools.call("report_candidate", {
            "keyframe_id": "K01/K01_V001/006", "confidence": .9, "reason": "visible subject"}, run, state.name)
        state.finish("done")
    monkeypatch.setattr(main.agent_service, "_run_agent", fake_agent)
    q = query(retrieval_database="btc", targets=[{"video_id": "K01_V001", "frame_idx": 500, "fps": 25}])
    rows = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://local") as client:
        for variant in "ABCDEF":
            snapshot, trace, wall = await run_one(client, q, variant, 5, .001)
            assert snapshot["finished"]
            assert snapshot["controller"]["status"] == "done"
            assert "trace" in trace and snapshot["ranking"]
            rows.append({"query_id": "q", "variant": variant, "metrics": measure(q, snapshot, trace["trace"], wall_s=wall)})
    write_report(tmp_path, rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary) == set("ABCDEF")
    assert summary["A"]["codex_calls"] == 0
    assert summary["B"]["codex_calls"] == 1 and summary["B"]["claude_calls"] == 0
    assert summary["C"]["codex_calls"] == summary["C"]["claude_calls"] == 1
    assert summary["D"]["jev_calls"] == 0  # mock mode cannot make paid calls
    assert summary["F"]["fallback"] == 1


def test_reranking_gain_is_not_an_agent_rescue():
    snap = {"policy": "rerank", "ranking": [f(probability=.95)], "baseline_ranking": [f(frame_idx=900)],
            "metrics": {"agent_calls": {}}, "controller": {"status": "done"}}
    metric = measure(query(), snap, [], wall_s=1)
    assert metric["improved_at_1"] and not metric["rescue1"]


def test_llm_judge_counts_as_system2_even_without_cli():
    snap = {"ranking": [f()], "controller": {"decision_calls": [{"backend": "llm", "cost_usd": .1,
             "input_tokens": 10, "output_tokens": 5}]}}
    metric = measure(query(), snap, [], wall_s=1)
    assert not metric["solved_without_system2"]
    assert metric["llm_judge_calls"] == 1


def test_unavailable_retrieval_is_not_an_ordinary_empty_result():
    trace = [{"kind": "retrieval", "warnings": ["visual channel unavailable: connection failed"]}]
    empty = measure(query(), {"baseline_ranking": []}, trace, wall_s=1)
    partial = measure(query(), {"baseline_ranking": [f()]}, trace, wall_s=1)
    assert empty["retrieval_failed"] and empty["retrieval_degraded"]
    assert partial["retrieval_degraded"] and not partial["retrieval_failed"]


@pytest.mark.parametrize("target", [
    {"video_id": "v", "frame_idx": float("nan")},
    {"video_id": "v", "start_s": 10, "end_s": 3},
    {"video_id": "v", "frame_idx": 4, "fps": 0},
    {"video_id": "v", "submit_keyframe_id": ""},
])
def test_dataset_rejects_invalid_moments(tmp_path, target):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(query(targets=[target])))
    with pytest.raises(ValueError):
        load_queries(path)


def test_dataset_rejects_cross_split_duplicate_and_unordered_sequence(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(query()) + "\n" + json.dumps(query(query_id="q2", query=" RED  BICYCLE ", split="dev")))
    with pytest.raises(ValueError, match="both dev and test"):
        load_queries(path)
    path.write_text(json.dumps(query(query_type="TRAKE", sequences=[[{"video_id": "v", "frame_idx": 200}, {"video_id": "v", "frame_idx": 100}]])))
    with pytest.raises(ValueError, match="increasing"):
        load_queries(path)


def test_dataset_rejects_temporal_task_disguised_as_kis(tmp_path):
    path = tmp_path / 'q.jsonl'
    path.write_text(json.dumps(query(query='E1: plate. E2: decorate.')))
    with pytest.raises(ValueError, match='TRAKE event-level'):
        load_queries(path)
    path.write_text(json.dumps(query(query_type='TRAKE', query='E1: plate. E2: decorate. E3: serve.',
        sequences=[[{'video_id': 'v', 'frame_idx': 1}, {'video_id': 'v', 'frame_idx': 2}]])))
    with pytest.raises(ValueError, match='every marked event'):
        load_queries(path)


def test_export_audits_mislabeled_temporal_query(monkeypatch):
    from types import SimpleNamespace
    from benchmarks.agent import export_dataset as module
    q = SimpleNamespace(query_id='TRAKE', text='E1: plate. E2: decorate.', gt=[('L26_V200', 4501)])
    monkeypatch.setattr(module, 'load_tkis', lambda: ([q], {}))
    monkeypatch.setattr(module, 'load_qa', lambda: [])
    monkeypatch.setattr(module, 'load_trake', lambda: [])
    rows, audit = module.export()
    assert not rows
    assert audit['skipped'][0]['query_id'] == 'TRAKE'
    assert 'event-to-frame' in audit['skipped'][0]['reason']


def test_paired_report_excludes_incomplete_queries_and_shows_agent_failures(tmp_path):
    from benchmarks.agent.run_benchmark import write_report
    rows = []
    for qid, variant, calls in [('q1', 'A', {}), ('q1', 'F', {'codex': 1, 'claude': 1}), ('q2', 'A', {})]:
        snapshot = {'ranking': [f()], 'metrics': {'agent_calls': calls},
                    'agents': {'codex': {'status': 'failed', 'error_kind': 'quota'}} if calls else {}}
        rows.append({'query_id': qid, 'variant': variant, 'metrics': measure(query(), snapshot, [], wall_s=1)})
    (tmp_path / 'manifest.json').write_text(json.dumps({'config': {'variants': ['A', 'F']}}))
    write_report(tmp_path, rows)
    paired = json.loads((tmp_path / 'summary_paired.json').read_text())
    assert paired['query_ids'] == ['q1']
    assert paired['summary']['A']['queries'] == paired['summary']['F']['queries'] == 1
    assert paired['summary']['F']['both_agents_rate'] == 1
    assert paired['summary']['F']['agent_failed'] == 1
    assert 'Agent failed' in (tmp_path / 'REPORT.md').read_text()


@pytest.mark.asyncio
async def test_runner_saves_quota_failure_then_stops_before_next_paid_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import httpx
    from benchmarks.agent import run_benchmark as module
    path = tmp_path / 'dataset.jsonl'
    path.write_text(json.dumps(query()))
    async def get(self, url):
        return httpx.Response(200, json={'mode': 'mock'}, request=httpx.Request('GET', 'http://local' + url))
    monkeypatch.setattr(httpx.AsyncClient, 'get', get)
    calls = []
    async def run(*args):
        calls.append(args[2])
        return {'controller': {'status': 'done'}, 'agents': {'codex': {'status': 'failed', 'error_kind': 'quota'}},
                'metrics': {'agent_calls': {'codex': 1}}}, {'trace': []}, 1
    monkeypatch.setattr(module, 'run_one', run)
    output = tmp_path / 'output'
    args = SimpleNamespace(dataset=path, output=output, split='test', limit=None, variants='B,C',
                           url='http://local', timeout=10, poll=.01, tolerance=1, seed=2026)
    with pytest.raises(SystemExit, match='quota exhausted'):
        await module.main_async(args)
    assert len(calls) == 1
    assert len((output / 'results.jsonl').read_text().splitlines()) == 1
    assert json.loads((output / 'STOPPED.json').read_text())['reason'] == 'agent_quota'
