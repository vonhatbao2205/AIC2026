"""Independent information-level evaluation, with no paid models or VBS state."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.agent.hints import build_plan, hint_stages, run_key, validate_hints
from benchmarks.agent.hint_report import write_hint_report
from benchmarks.agent.metrics import measure
from benchmarks.agent.paper_report import export_paper_report, paired_intervals
from benchmarks.agent.run_benchmark import load_queries, run_one, write_report


def query(qid="q", hints=None, **extra):
    hints = hints or ["Xe đạp màu đỏ.", "Một người đang sửa xe.", "Bên cạnh cửa màu xanh."]
    return {"query_id": qid, "query": " ".join(hints), "hints_vi": hints,
            "query_type": "T-KIS", "split": "test", "retrieval_database": "btc",
            "targets": [{"video_id": "K01_V001", "frame_idx": 100, "fps": 25}], **extra}


def frame(**extra):
    return {"video_id": "K01_V001", "frame_idx": 100, "fps": 25, "pts_time": 4, **extra}


def row(stage, variant, hit=False, video=False, **extra):
    ranking = [frame(frame_idx=100 if hit else 999)] if hit or video else []
    return {**{k: stage[k] for k in ("query_id", "query_type", "hint_stage", "hint_level", "hint_count", "hint_fraction")},
            "base_query_id": stage["query_id"], "variant": variant, "split": stage["split"],
            "metrics": measure(stage, {"ranking": ranking, "controller": {"status": "done"}, **extra}, [], wall_s=2)}


def args(path, output, **extra):
    return SimpleNamespace(dataset=path, output=output, split="test", limit=None, variants=None,
                           url="http://local", timeout=5, poll=.001, tolerance=1, seed=2026,
                           hint_mode="cumulative", dry_run=False, **extra)


def test_prefix_construction_preserves_reviewed_vietnamese_and_full_once():
    q = query(previous_hints=["old"], retrieval_id="stale")
    before = copy.deepcopy(q)
    stages = hint_stages(q, "cumulative")
    assert q == before
    assert [s["hint_stage"] for s in stages] == ["h1", "h2", "full"]
    assert [s["hint_level"] for s in stages] == [1, 2, 3]
    assert stages[1]["query"] == "\n".join(q["hints_vi"][:2])
    assert stages[1]["hint_fraction"] == pytest.approx(2 / 3)
    assert all(s["previous_hints"] == [] and s["base_query_id"] == "q" for s in stages)
    assert hint_stages(q, "full")[0]["query"] == q["query"]
    stages[0]["targets"][0]["video_id"] = "mutation"
    assert stages[1]["targets"] == q["targets"]


@pytest.mark.parametrize("hints", ["not a list", [""], ["valid", None], ["changed facts"]])
def test_malformed_or_mismatched_hints_fail_before_model_calls(tmp_path, hints):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({**query(), "hints_vi": hints}))
    with pytest.raises(ValueError, match="hints_vi"):
        load_queries(path)


def test_only_formatting_differences_are_accepted():
    q = query()
    q["query"] = q["query"].replace(" ", "  ")
    validate_hints(q)
    q["query"] = q["query"].replace("đỏ", "vàng")
    with pytest.raises(ValueError, match="audit"):
        validate_hints(q)


def test_subset_exclusions_are_audited_without_fabricated_hint_boundaries():
    plain = {**query("plain"), "hints_vi": []}
    single = query("single", ["One hint"])
    unsupported = query("temporal", query_type="TRAKE")
    stages, plan = build_plan([query(), plain, single, unsupported], "cumulative", ["A", "C", "F"])
    assert len(stages) == 3 and plan["planned_runs"] == 9
    assert {q["query_id"] for q in plan["excluded"]} == {"plain", "single", "temporal"}
    full, _ = build_plan([plain], "full", ["A"])
    assert full[0]["hint_level"] is None and full[0]["query"] == plain["query"]
    limited, plan = build_plan([plain, query("first"), query("second")], "cumulative", ["A"], limit=1)
    assert {q["query_id"] for q in limited} == {"first"} and plan["planned_runs"] == 3


@pytest.mark.asyncio
async def test_each_level_payload_contains_only_current_query_and_empty_previous_hints():
    bodies = []
    def respond(request):
        if request.method == "POST":
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"run_id": str(len(bodies)), "finished": True})
        return httpx.Response(200, json={"trace": []})
    q = query(previous_hints=["leaked history"], retrieval_id="seed", replaces="past-run", candidates=[frame()])
    async with httpx.AsyncClient(base_url="http://local", transport=httpx.MockTransport(respond)) as client:
        for stage in hint_stages(q, "cumulative"):
            await run_one(client, stage, "F", 1, .001)
    assert [b["query"] for b in bodies] == [s["query"] for s in hint_stages(q, "cumulative")]
    for body in bodies:
        assert body["previous_hints"] == []
        assert not ({"retrieval_id", "replaces", "targets", "candidates", "hints_vi", "hint_count", "sequences"} & body.keys())


@pytest.mark.asyncio
async def test_real_backend_constructs_fresh_board_controller_history_and_cache(monkeypatch):
    from app import main
    from app.agent.controller import Controller
    controllers = []
    original = Controller.__init__
    def capture(self, *a, **kw):
        original(self, *a, **kw)
        run = self.run
        assert run.request.previous_hints == []
        assert not run.board.candidates and not run.candidates and not run.trace
        assert not run.tool_cache and not run.tool_locks
        assert not self.invoked and not self.actions and not self.client.calls
        assert all(not agent.steps for agent in run.agents.values())
        controllers.append(self)
    monkeypatch.setattr(Controller, "__init__", capture)
    monkeypatch.setattr(main.agent_service, "_binary", lambda name: "fake")
    async def fake_agent(run, state, binary, url):
        state.begin()
        run.tool_cache["sentinel"] = run.request.query
        state.finish("done")
    monkeypatch.setattr(main.agent_service, "_run_agent", fake_agent)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://local") as client:
        # Reverse order demonstrates that levels have no sequential dependency.
        for stage in reversed(hint_stages(query(), "cumulative")):
            snapshot, _, _ = await run_one(client, stage, "F", 5, .001)
            assert snapshot["finished"]
    assert len({c.run.id for c in controllers}) == 3
    assert len({id(c.run.board) for c in controllers}) == 3
    assert len({id(c.client) for c in controllers}) == 3
    assert len({id(c.run.tool_cache) for c in controllers}) == 3


def test_qa_video_moment_and_strict_success_are_distinct():
    q = query(query_type="QA", targets=[{"video_id": "K01_V001", "frame_idx": 100, "answers": ["4"]}])
    m = measure(q, {"ranking": [frame(answer="wrong")]}, [], wall_s=1)
    assert m["video_rank"] == 1 and m["moment_r1"] == 1 and m["r1"] == 0


def test_solve_metrics_handle_unsolved_nonmonotonic_and_incomplete_runs(tmp_path):
    rows = []
    for qid in ["early", "late", "unsolved"]:
        for stage in hint_stages(query(qid), "cumulative"):
            for v in ["A", "F"]:
                hit = (qid == "early" and stage["hint_level"] == 1) or (qid == "late" and stage["hint_level"] == 3)
                rows.append(row(stage, v, hit=hit))
    for stage in hint_stages(query("incomplete"), "cumulative"):
        rows.append(row(stage, "A", hit=True))
    report = write_hint_report(tmp_path, rows, ["A", "F"])
    solve = report["solve"]
    assert solve["complete_base_queries"] == 3 and solve["planned_base_queries"] == 4
    s = solve["summary"]["F"]["strict"]
    assert s["solved_queries"] == 2 and s["unsolved_queries"] == 1
    assert s["mean_hints_to_solve_solved_only"] == 2
    assert s["mean_hints_to_solve_penalized"] == pytest.approx(8 / 3)
    assert s["early_solve_rate"] == s["mean_hint_saving"] == s["regression_rate"] == pytest.approx(1 / 3)
    assert solve["query_ids"] == ["early", "late", "unsolved"]
    assert report["levels"]["h1"]["paired"]["summary"]["F"]["queries"] == 3
    assert (tmp_path / "hint_results.csv").exists()


def test_main_table_never_pools_hint_levels_and_strata_have_explicit_n(tmp_path):
    rows = []
    for q in [query("three"), query("four", ["a", "b", "c", "d"])]:
        for stage in hint_stages(q, "cumulative"):
            rows.append(row(stage, "A", hit=stage["hint_stage"] == "full"))
    write_report(tmp_path, rows)
    summary = json.loads((tmp_path / "summary.json").read_text())["A"]
    assert summary["queries"] == 2 and summary["r1"] == 1
    hints = json.loads((tmp_path / "summary_by_hint.json").read_text())
    assert hints["levels"]["h3"]["summary"]["A"]["queries"] == 1
    assert hints["levels"]["full"]["summary"]["A"]["queries"] == 2
    assert set(hints["by_total_hint_count"]) == {"3", "4"}


@pytest.mark.asyncio
async def test_dry_run_writes_plan_without_network(tmp_path, monkeypatch):
    from benchmarks.agent import run_benchmark as module
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(query()))
    def forbidden(*a, **kw):
        raise AssertionError("dry run accessed HTTP")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    a = args(path, tmp_path / "output")
    a.dry_run = True
    await module.main_async(a)
    plan = json.loads((a.output / "run_plan.json").read_text())
    assert plan["planned_runs"] == 9 and plan["variants"] == ["A", "C", "F"]
    assert not (a.output / "manifest.json").exists()


@pytest.mark.asyncio
async def test_resume_is_per_level_and_changed_protocol_cannot_resume(tmp_path, monkeypatch):
    from benchmarks.agent import run_benchmark as module
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(query()))
    async def get(self, url):
        return httpx.Response(200, json={"mode": "mock"}, request=httpx.Request("GET", "http://local" + url))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    calls = []
    async def fake(client, stage, variant, timeout, poll):
        calls.append((stage["hint_stage"], variant))
        return {"finished": True, "ranking": [frame()], "controller": {"status": "done"}}, {"trace": []}, 1
    monkeypatch.setattr(module, "run_one", fake)
    a = args(path, tmp_path / "output")
    a.variants = "A,F"
    await module.main_async(a)
    assert len(calls) == 6
    results = a.output / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len({run_key(r) for r in rows}) == 6
    results.write_text("\n".join(json.dumps(r) for r in rows[:-1]) + "\n")
    await module.main_async(a)
    assert len(calls) == 7
    await module.main_async(a)
    assert len(calls) == 7
    a.hint_mode = "full"
    with pytest.raises(ValueError, match="plan changed"):
        await module.main_async(a)


@pytest.mark.asyncio
async def test_cumulative_http_failure_is_in_stage_denominator(tmp_path, monkeypatch):
    from benchmarks.agent import run_benchmark as module
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(query()))
    async def get(self, url):
        return httpx.Response(200, json={"mode": "mock"}, request=httpx.Request("GET", "http://local" + url))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    async def fail(*a):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(module, "run_one", fail)
    a = args(path, tmp_path / "output")
    a.variants = "F"
    await module.main_async(a)
    summary = json.loads((a.output / "summary_by_hint.json").read_text())
    assert summary["levels"]["h1"]["summary"]["F"]["failed"] == 1
    assert summary["solve"]["summary"]["F"]["strict"]["unsolved_queries"] == 1


def test_paired_bootstrap_and_offline_paper_export(tmp_path):
    rows = []
    for stage in hint_stages(query(), "cumulative"):
        for v in ["A", "F"]:
            rows.append(row(stage, v, hit=v == "F"))
    full = [r for r in rows if r["hint_stage"] == "full"]
    ci = paired_intervals(full, ["A", "F"], samples=30)
    assert ci["paired_differences"]["F-A"]["r1"]["ci95"] == [1, 1]
    assert paired_intervals([full[0]], ["A", "F"], samples=10)["query_ids"] == []
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"variants": ["A", "F"], "hint_mode": "cumulative"}}))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = export_paper_report(tmp_path, samples=30)
    assert set(result["hint_levels"]) == {"h1", "h2", "full"}
    assert (tmp_path / "paper_tables.csv").exists() and (tmp_path / "PAPER_REPORT.md").exists()


def test_exportable_hint_figures_use_complete_fixed_cohorts(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.agent.paper_report import plot_hint_curves
    rows = [row(s, v, hit=v == "F") for q in [query(), query("four", ["a", "b", "c", "d"])]
            for s in hint_stages(q, "cumulative") for v in ["A", "F"]]
    write_hint_report(tmp_path, rows, ["A", "F"])
    files = plot_hint_curves(tmp_path, rows, ["A", "F"])
    assert set(files) == {"hint_recall.pdf", "hint_recall.png", "hint_computation.pdf", "hint_computation.png"}
    assert all((tmp_path / name).stat().st_size > 1000 for name in files)


@pytest.mark.asyncio
async def test_soict_suite_dry_run_and_quota_stop(tmp_path, monkeypatch):
    from benchmarks.agent import run_soict as suite
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(query()))
    a = args(path, tmp_path / "output")
    a.experiment, a.dry_run = "all", True
    await suite.run_suite(a)
    plan = json.loads((a.output / "suite-test.json").read_text())
    assert plan["planned_runs"] == 15
    assert (a.output / "main-test/run_plan.json").exists()
    assert (a.output / "cumulative-test/run_plan.json").exists()
    calls = []
    async def quota(options):
        calls.append(options.hint_mode)
        raise SystemExit("quota")
    monkeypatch.setattr(suite, "main_async", quota)
    with pytest.raises(SystemExit, match="quota"):
        await suite.run_suite(a)
    assert calls == ["full"]
    a.experiment = "ablations"
    with pytest.raises(ValueError, match="dev"):
        await suite.run_suite(a)


def test_merge_keeps_reviewed_hint_boundaries_and_rejects_conflicts():
    from benchmarks.agent.export_dataset import merge_additional
    base = query("old")
    base.pop("hints_vi")
    incoming = query("new")
    merged, _ = merge_additional([base], [incoming])
    assert merged[0]["hints_vi"] == incoming["hints_vi"]
    conflict = query("conflict", [base["query"]])
    with pytest.raises(ValueError, match="hint boundaries"):
        merge_additional(merged, [conflict])
