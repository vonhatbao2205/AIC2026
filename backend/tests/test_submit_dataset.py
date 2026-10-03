"""Reviewed contest submissions must not silently become unaudited labels."""
import json

import pytest

from benchmarks.agent.export_dataset import group_video_splits, merge_additional
from benchmarks.agent.export_submit_dataset import digest, export_review, target_from_submission


def bundle(tmp_path, *, kind="QA", verdict="WRONG", status="dres_ok"):
    submission = {"id": "s1", "status": status, "verdict": verdict, "query_type": kind,
                  "payload": {"video_id": "N025-V002", "frame_idx": 100,
                              "timestamp": 4.2, "fps": 25, "answer": "4"}}
    queue = [{"query_id": "q1", "query": "How many wings?", "query_type": kind,
              "hints_vi": [], "workbook_rows": [1], "submissions": [submission],
              "selection_basis": "owner_confirmed_btc_label_error"}]
    evidence = [{"directory": "evidence/row-01", "moments": [{"submission_id": "s1",
                 "video_id": "N025-V002", "event_index": 0, "submitted_time_s": 4.2}]}]
    documents = {"review_queue": queue, "source_manifest": {"source": "test"},
                 "evidence_index": evidence, "evidence_manifest": {"artifacts": []}}
    for name, content in documents.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content))
    review = {f"{name}_sha256": digest(tmp_path / f"{name}.json") for name in documents}
    review.update({"owner_authorization": "Organizer error confirmed by owner", "limitations": [],
                   "decisions": [{"query_id": "q1", "review_status": "reviewed",
                                  "accepted_submission_ids": ["s1"], "observation": "Ingredient card shows four wings",
                                  "evidence_directory": "evidence/row-01", "method": "sampled_frames_from_decoded_video",
                                  "label_authority": "owner_confirmed_btc_label_error"}]})
    (tmp_path / "review_decisions.json").write_text(json.dumps(review))
    return queue, review


def rewrite(tmp_path, name, value, review):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(value))
    review[f"{name}_sha256"] = digest(path)
    (tmp_path / "review_decisions.json").write_text(json.dumps(review))


def test_reviewed_wrong_override_retains_original_verdict_and_pts(tmp_path):
    bundle(tmp_path)
    rows, audit = export_review(tmp_path)
    assert rows[0]["targets"][0]["start_s"] == 4.2  # not 100 / 25
    assert rows[0]["targets"][0]["answers"] == ["4"]
    assert rows[0]["provenance"]["original_dres_verdicts"] == {"s1": "WRONG"}
    assert rows[0]["ground_truth_source"] == "owner_confirmed_btc_label_error"
    assert audit["by_label_authority"]["owner_confirmed_btc_label_error"] == 1


@pytest.mark.parametrize("verdict,status", [(None, "dres_error"), ("CORRECT", "dres_ok")])
def test_wrong_only_override_never_accepts_errors_or_a_correct_task(tmp_path, verdict, status):
    bundle(tmp_path, verdict=verdict, status=status)
    with pytest.raises(ValueError, match="override|Overrides"):
        export_review(tmp_path)


def test_missing_owner_authorization_rejects_wrong_override(tmp_path):
    _, review = bundle(tmp_path)
    review.pop("owner_authorization")
    (tmp_path / "review_decisions.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="owner authorization"):
        export_review(tmp_path)


def test_changed_query_requires_fresh_review(tmp_path):
    queue, _ = bundle(tmp_path)
    queue[0]["query"] = "How many legs?"
    (tmp_path / "review_queue.json").write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="stale"):
        export_review(tmp_path)


def test_additional_hint_audit_is_bound_to_the_review(tmp_path):
    _, review = bundle(tmp_path)
    path = tmp_path / "hint_audit.json"
    path.write_text(json.dumps({"hint_checks": ["supported"]}))
    review["decisions"][0].update({"additional_audit_reference": path.name,
                                   "additional_audit_sha256": digest(path)})
    (tmp_path / "review_decisions.json").write_text(json.dumps(review))
    rows, _ = export_review(tmp_path)
    assert rows[0]["provenance"]["additional_audit_sha256"] == digest(path)
    path.write_text(json.dumps({"hint_checks": ["refuted"]}))
    with pytest.raises(ValueError, match="stale"):
        export_review(tmp_path)


def test_visual_evidence_must_match_the_selected_timestamp(tmp_path):
    _, review = bundle(tmp_path)
    path = tmp_path / "evidence_index.json"
    evidence = json.loads(path.read_text())
    evidence[0]["moments"][0]["submitted_time_s"] = 40.2
    rewrite(tmp_path, "evidence_index", evidence, review)
    with pytest.raises(ValueError, match="timestamp"):
        export_review(tmp_path)


def test_single_frame_cannot_label_a_four_event_trake(tmp_path):
    queue, review = bundle(tmp_path, kind="TRAKE")
    queue[0]["query"] = "E1: wave. E2: heart. E3: overtake. E4: school."
    rewrite(tmp_path, "review_queue", queue, review)
    with pytest.raises(ValueError, match="every explicit event"):
        export_review(tmp_path)


def test_temporal_target_cannot_guess_fps():
    with pytest.raises(ValueError, match="timestamp"):
        target_from_submission({"query_type": "T-KIS", "payload": {"video_id": "v", "frame_idx": 100}})


def visual_bundle(tmp_path):
    queue, review = bundle(tmp_path, kind="TRAKE")
    query = queue[0]
    query.update({"query": "E1: wave. E2: heart.", "submissions": [],
                  "selection_basis": "pe_retrieval_visual_annotation",
                  "visual_annotations": [{"id": "annotation-1", "payload": {
                      "video_id": "S01-V011", "fps": 30,
                      "events": [{"event_index": 1, "pts_time": 4.2, "frame_idx": 126},
                                 {"event_index": 2, "pts_time": 5.1, "frame_idx": 153}]},
                      "provenance": {"event_boundaries": [
                          {"event_index": i, "selected_time_s": t, "criterion": criterion,
                           "evidence_frames": [f"event-{i}.jpg"], "uncertainty_note": "One frame"}
                          for i, t, criterion in [(1, 4.2, "Five fingers visible"), (2, 5.1, "First hand contact")]]}}]})
    review["visual_annotation_authorization"] = "Owner requested PE search and visual annotation in S01"
    review["decisions"][0].update({"accepted_submission_ids": [], "accepted_annotation_ids": ["annotation-1"],
                                   "label_authority": query["selection_basis"],
                                   "method": "pe_search_and_dense_video_frame_review"})
    rewrite(tmp_path, "review_queue", queue, review)
    evidence = [{"directory": "evidence/row-01", "moments": [
        {"annotation_id": "annotation-1", "video_id": "S01-V011", "event_index": i, "annotated_time_s": t}
        for i, t in [(1, 4.2), (2, 5.1)]]}]
    rewrite(tmp_path, "evidence_index", evidence, review)
    return queue, review


def test_pe_visual_annotation_exports_complete_sequence_without_fabricating_dres(tmp_path):
    visual_bundle(tmp_path)
    rows, _ = export_review(tmp_path)
    result = rows[0]
    assert [t["frame_idx"] for t in result["sequences"][0]] == [126, 153]
    assert result["provenance"]["original_dres_verdicts"] == {}
    assert result["provenance"]["accepted_submission_ids"] == []
    assert result["provenance"]["accepted_annotation_ids"] == ["annotation-1"]
    assert result["ground_truth_source"] == "pe_retrieval_visual_annotation"


@pytest.mark.parametrize("missing", ["authorization", "event_boundary", "boundary_timestamp"])
def test_visual_labels_require_authorization_and_matching_event_boundaries(tmp_path, missing):
    queue, review = visual_bundle(tmp_path)
    if missing == "authorization":
        review.pop("visual_annotation_authorization")
        (tmp_path / "review_decisions.json").write_text(json.dumps(review))
    else:
        boundaries = queue[0]["visual_annotations"][0]["provenance"]["event_boundaries"]
        if missing == "event_boundary":
            boundaries.pop()
        else:
            boundaries[0]["selected_time_s"] = 99
        rewrite(tmp_path, "review_queue", queue, review)
    with pytest.raises(ValueError, match="authorization|boundar"):
        export_review(tmp_path)


def row(qid, split, videos, text=None):
    return {"query_id": qid, "query": text or qid, "query_type": "T-KIS", "split": split,
            "retrieval_database": "infoshotpp", "targets": [{"video_id": v, "frame_idx": 1} for v in videos]}


def test_merging_repeated_query_keeps_alternatives_without_duplicate_questions():
    base = [row("old", "dev", ["v1"], "A cyclist waves")]
    extra = [row("new", "test", ["v2"], "A CYCLIST waves")]
    merged, audit = merge_additional(base, extra)
    assert len(merged) == 1 and merged[0]["split"] == "dev"
    assert {t["video_id"] for t in merged[0]["targets"]} == {"v1", "v2"}
    assert len(base[0]["targets"]) == 1
    assert audit["exact_duplicates_merged"] == [{"kept": "old", "merged": "new"}]


def test_split_grouping_blocks_transitive_video_leakage():
    base = [row("a", "dev", ["v1"]), row("b", "test", ["v1", "v2"]), row("c", "test", ["v2"]), row("d", "test", ["v3"])]
    grouped, audit = group_video_splits(base)
    assert [q["split"] for q in grouped] == ["dev", "dev", "dev", "test"]
    assert len(audit["changes"]) == 2
    assert base[1]["split"] == "test"


def test_added_dev_labels_never_move_to_test_on_merge():
    base = [row("old", "test", ["v1"])]
    extra = [row("new", "dev", ["v1"])]
    merged, _ = merge_additional(base, extra)
    assert merged[1]["split"] == "dev"
    grouped, _ = group_video_splits(merged)
    assert all(q["split"] == "dev" for q in grouped)


def test_repeated_dev_question_promotes_existing_test_without_mutating_base():
    base = [row("old", "test", ["v1"], "A cyclist waves")]
    extra = [row("new", "dev", ["v2"], "A cyclist waves")]
    merged, _ = merge_additional(base, extra)
    assert merged[0]["split"] == "dev"
    assert base[0]["split"] == "test" and "additional_ground_truth_sources" not in base[0]
