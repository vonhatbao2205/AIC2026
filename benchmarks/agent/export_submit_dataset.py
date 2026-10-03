"""Export explicitly reviewed workbook/submission matches as agent benchmark labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from .run_benchmark import load_queries, validate_target
from backend.app.trake_events import split_marked_events

DEFAULT_REVIEW = Path(__file__).parent / "annotations" / "aic26-20260926"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def target_from_submission(submission: dict, event: dict | None = None) -> dict:
    payload = submission["payload"]
    point = event if event is not None else payload
    timestamp = point.get("pts_time") if event is not None else point.get("timestamp")
    fps = payload.get("fps")
    if timestamp is None and point.get("frame_idx") is not None and fps:
        timestamp = point["frame_idx"] / fps
    if timestamp is None:
        raise ValueError("A submitted point requires timestamp or frame_idx with FPS")
    # PTS is the primary coordinate. N camera streams can be variable-frame-rate;
    # never replace their recorded PTS with frame_idx / average FPS.
    target = {"video_id": point.get("video_id") or payload["video_id"],
              "start_s": timestamp, "end_s": timestamp}
    if point.get("frame_idx") is not None:
        target["frame_idx"] = point["frame_idx"]
    if fps:
        target["fps"] = fps
    if submission["query_type"] == "QA":
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("QA ground truth requires the reviewed submitted answer")
        target["answers"] = [answer.strip()]
    validate_target(target)
    return target


def export_review(review_dir: Path):
    queue_path, source_path = review_dir / "review_queue.json", review_dir / "source_manifest.json"
    decision_path = review_dir / "review_decisions.json"
    queue = json.loads(queue_path.read_text())
    review = json.loads(decision_path.read_text())
    index_path = review_dir / "evidence_index.json"
    media_manifest_path = review_dir / "evidence_manifest.json"
    for name, path in (("review_queue", queue_path), ("source_manifest", source_path),
                       ("evidence_index", index_path), ("evidence_manifest", media_manifest_path)):
        if review.get(f"{name}_sha256") != digest(path):
            raise ValueError(f"Review is stale: {name} changed")
    evidence = {entry["directory"]: entry for entry in json.loads(index_path.read_text())}
    decisions = {d["query_id"]: d for d in review["decisions"]}
    if len(decisions) != len(review["decisions"]) or set(decisions) != {q["query_id"] for q in queue}:
        raise ValueError("Every workbook query needs one explicit review decision")
    rows, skipped = [], []
    for query in queue:
        decision = decisions[query["query_id"]]
        if decision["review_status"] != "reviewed":
            skipped.append({"query_id": query["query_id"], "workbook_rows": query["workbook_rows"],
                            "reason": decision["observation"]})
            continue
        if decision.get("additional_audit_reference"):
            audit_path = (review_dir / decision["additional_audit_reference"]).resolve()
            if not audit_path.is_relative_to(review_dir.resolve()) or decision.get("additional_audit_sha256") != digest(audit_path):
                raise ValueError("Review is stale: additional audit changed or is outside the review directory")
        visual_annotation = decision["label_authority"] == "pe_retrieval_visual_annotation"
        accepted_ids = decision.get("accepted_annotation_ids", []) if visual_annotation else decision["accepted_submission_ids"]
        if visual_annotation:
            # An annotation is a local label, never a fabricated DRES submission.
            submissions = {a["id"]: {"id": a["id"], "query_type": query["query_type"], "payload": a["payload"],
                                       "annotation_provenance": a["provenance"]}
                           for a in query.get("visual_annotations", [])}
        else:
            submissions = {s["id"]: s for s in query["submissions"]}
        if not accepted_ids or len(set(accepted_ids)) != len(accepted_ids) or not set(accepted_ids) <= submissions.keys():
            raise ValueError("Review must select existing, unique submission IDs")
        if not decision.get("observation") or not decision.get("evidence_directory"):
            raise ValueError("Reviewed labels require visual observations and evidence references")
        if decision["label_authority"] != query["selection_basis"]:
            raise ValueError("Review authority disagrees with the candidate selection basis")
        selected = [submissions[i] for i in accepted_ids]
        observed = evidence.get(decision["evidence_directory"], {}).get("moments", [])
        for submission in selected:
            moments = [m for m in observed if (m.get("annotation_id") or m.get("submission_id")) == submission["id"]]
            expected_events = submission["payload"].get("events") or [submission["payload"]]
            if len(moments) != len(expected_events):
                raise ValueError("Every selected submitted moment must have decoded visual evidence")
            for event in expected_events:
                target = target_from_submission(submission, event if submission["payload"].get("events") else None)
                if not any(m["video_id"] == target["video_id"] and
                           m["event_index"] == event.get("event_index", 0) and
                           abs(m.get("annotated_time_s", m.get("submitted_time_s", -1)) - target["start_s"]) < .001 for m in moments):
                    raise ValueError("Evidence does not match the selected video/event/timestamp")
        if decision["label_authority"] == "dres_correct":
            if any(s["status"] != "dres_ok" or s["verdict"] != "CORRECT" for s in selected):
                raise ValueError("DRES-correct labels must retain the original CORRECT verdict")
        elif decision["label_authority"] == "owner_confirmed_btc_label_error":
            if not review.get("owner_authorization") or any(s["verdict"] == "CORRECT" for s in submissions.values()):
                raise ValueError("Wrong-only override needs owner authorization and no CORRECT submission")
            if any(s["status"] != "dres_ok" or s["verdict"] != "WRONG" for s in selected):
                raise ValueError("Overrides can select reviewed WRONG submissions, never DRES errors")
        elif visual_annotation:
            if not review.get("visual_annotation_authorization") or query["query_type"] != "TRAKE":
                raise ValueError("PE visual annotation requires owner authorization and a TRAKE query")
            if decision["method"] != "pe_search_and_dense_video_frame_review":
                raise ValueError("Visual TRAKE annotation needs dense event/frame inspection")
            for annotation in selected:
                boundaries = annotation["annotation_provenance"].get("event_boundaries", [])
                events = annotation["payload"].get("events", [])
                if len(boundaries) != len(events) or not events:
                    raise ValueError("Visual TRAKE annotation must record every event boundary decision")
                for event, boundary in zip(events, boundaries):
                    timestamp = target_from_submission(annotation, event)["start_s"]
                    if (boundary.get("event_index") != event.get("event_index") or
                            not boundary.get("criterion") or not boundary.get("evidence_frames") or
                            not boundary.get("uncertainty_note") or
                            abs(boundary.get("selected_time_s", -1) - timestamp) > .001):
                        raise ValueError("Visual event boundaries must match the annotated moments and record evidence/uncertainty")
        else:
            raise ValueError("Unknown label authority")
        targets, sequences = [], []
        for submission in selected:
            if query["query_type"] == "TRAKE":
                events = submission["payload"].get("events") or []
                expected = len(split_marked_events(query["query"]))
                if expected < 2 or len(events) != expected or [e.get("event_index") for e in events] != list(range(1, expected + 1)):
                    raise ValueError("TRAKE requires every explicit event, in order; a single frame is insufficient")
                sequence = [target_from_submission(submission, event) for event in events]
                if len({t["video_id"] for t in sequence}) != 1 or any(a["start_s"] >= b["start_s"] for a, b in zip(sequence, sequence[1:])):
                    raise ValueError("TRAKE targets must be ordered moments in one video")
                if sequence not in sequences:
                    sequences.append(sequence)
                for target in sequence:
                    if target not in targets:
                        targets.append(target)
            else:
                target = target_from_submission(submission)
                if target not in targets:
                    targets.append(target)
        split = "dev" if int(hashlib.sha256(text_key(query["query"]).encode()).hexdigest()[:8], 16) % 5 == 0 else "test"
        rows.append({"query_id": query["query_id"], "query": query["query"], "query_type": query["query_type"],
                     "split": split, "retrieval_database": "infoshotpp", "image_models": ["pe"],
                     "targets": targets, **({"sequences": sequences} if sequences else {}),
                     "hints_vi": query["hints_vi"], "workbook_rows": query["workbook_rows"],
                     "ground_truth_source": decision["label_authority"],
                     "provenance": {"source_manifest_sha256": digest(source_path), "review_decisions_sha256": digest(decision_path),
                                    "accepted_submission_ids": [] if visual_annotation else accepted_ids,
                                    "accepted_annotation_ids": accepted_ids if visual_annotation else [],
                                    "original_dres_verdicts": {s["id"]: s["verdict"] for s in selected if "verdict" in s},
                                    "review_method": decision["method"], "observation": decision["observation"],
                                    "evidence_directory": decision["evidence_directory"],
                                    "annotation_kind": "video_annotated_event_points" if visual_annotation else "submitted_points_with_brief_visual_review",
                                    **({"additional_audit_reference": decision["additional_audit_reference"],
                                        "additional_audit_sha256": decision["additional_audit_sha256"]}
                                       if decision.get("additional_audit_reference") else {}),
                                    **({"annotation_details": [s["annotation_provenance"] for s in selected]} if visual_annotation else {}),
                                    "complete_official_acceptance_intervals": False}})
    audit = {"queries": len(rows), "by_task": dict(Counter(r["query_type"] for r in rows)),
             "by_label_authority": dict(Counter(r["ground_truth_source"] for r in rows)),
             "by_split": dict(Counter(r["split"] for r in rows)), "skipped": skipped,
             "limitations": review["limitations"], "source_manifest_sha256": digest(source_path),
             "review_decisions_sha256": digest(decision_path)}
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-media", action="store_true", help="Check local evidence file sizes and SHA-256")
    args = parser.parse_args()
    rows, audit = export_review(args.review_dir)
    if args.verify_media:
        media = json.loads((args.review_dir / "evidence_manifest.json").read_text())
        for artifact in media["artifacts"]:
            path = args.review_dir / artifact["path"]
            if path.stat().st_size != artifact["size_bytes"] or digest(path) != artifact["sha256"]:
                raise ValueError(f"Media evidence changed: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if args.output.exists() and args.output.read_text() != content:
        raise SystemExit("Output contains a different dataset; choose a new version/path")
    args.output.write_text(content)
    load_queries(args.output)
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: audit[k] for k in ("queries", "by_task", "by_label_authority", "by_split")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
