"""Apply content-bound owner approval without changing evidence provenance.

Approvals are explicit records, never inferred from an exporter or validator.
Changing a query, interval or hint invalidates its recorded approval digest.
"""
import copy
import hashlib
import json
from pathlib import Path


def content_digest(record):
    content = {
        "query_id": record["query_id"],
        "original_query": record["original_query"],
        "task_type": record["task_type"],
        "time_unit": record["time_unit"],
        "interval_convention": record["interval_convention"],
        "targets": [{
            "video_id": target["video_id"],
            "duration_ms": target["duration_ms"],
            "candidate_intervals": target["candidate_intervals"],
            "context_intervals": target["context_intervals"],
        } for target in record["targets"]],
        "hints": [{key: hint[key] for key in (
            "hint_id", "delta_text", "cumulative_text", "evidence_ids", "source",
        )} for hint in record["hints"]],
        "duplicate_representative_query_id": record.get("duplicate_representative_query_id"),
    }
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def apply_owner_approval(records, path=None):
    path = Path(path) if path else Path(__file__).with_name("owner_approval.json")
    if not path.exists():
        return records
    approval = json.loads(path.read_text())
    for record in records:
        expected = approval["record_digests"].get(record["query_id"])
        if expected is None:
            continue
        if content_digest(record) != expected:
            raise ValueError(f"Annotation changed since owner approval: {record['query_id']}")
        if len(record["hints"]) != 3 or not any(t["candidate_intervals"] for t in record["targets"]):
            raise ValueError(f"Approved record lacks intervals or three hints: {record['query_id']}")
        record.update(
            annotation_status="source_normalized",
            review_status="approved",
            reviewer="dataset_owner",
            review_basis="owner_confirmation_of_normalized_existing_ground_truth",
            approved_at=approval["at"],
            approval_id=approval["approval_id"],
            approval_content_sha256=expected,
        )
        representative = record.get("duplicate_representative_query_id")
        reason = ("non_tkis_task" if record["task_type"] != "TKIS" else
                  "exact_duplicate_non_representative" if representative and representative != record["query_id"] else None)
        record["eligible_for_benchmark"] = reason is None
        record["exclusion_reason"] = reason
        for target in record["targets"]:
            target["target_intervals"] = []
            for proposed in target["candidate_intervals"]:
                accepted = copy.deepcopy(proposed)
                accepted.update(status="accepted", interval_role="answer", approval_id=approval["approval_id"],
                                acceptance_basis=record["review_basis"])
                accepted["justification"] = (
                    "Dataset owner accepted normalized intervals for benchmark use. "
                    "Original sampling and boundary limitations remain in candidate_intervals and evidence."
                )
                target["target_intervals"].append(accepted)
        for hint in record["hints"]:
            hint["review_status"] = "approved"
            hint["approval_id"] = approval["approval_id"]
        for issue in record["issues"]:
            if issue["code"] not in {"non_tkis_task", "exact_duplicate"}:
                issue["benchmark_blocking"] = False
                issue["disposition"] = "retained_provenance_limitation; owner_accepted_for_benchmark"
    return records
