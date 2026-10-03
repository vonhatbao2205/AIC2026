"""Export existing workbook labels without inventing missing ground truth."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import unicodedata
from pathlib import Path

from benchmarks.gt_loader import load_qa, load_tkis, load_trake
from backend.app.trake_events import split_marked_events
from benchmarks.agent.run_benchmark import load_queries
from benchmarks.agent.hints import validate_hints


def profile(video):
    return "btc" if video.startswith("K") else "infoshotpp"


def export():
    rows, skipped = [], []
    tkis, reasons = load_tkis()
    for kind, queries in (("T-KIS", tkis), ("QA", load_qa()), ("TRAKE", load_trake())):
        for q in queries:
            events = split_marked_events(q.text)
            if kind != "TRAKE" and events:
                skipped.append({"query_id": q.query_id, "type": kind,
                                "reason": "marked temporal query lacks event-to-frame labels in this workbook"})
                continue
            sequences = []
            if kind == "TRAKE":
                for alternative in q.alternatives.values():
                    if len(alternative) < 2 or [e for e, _, _ in alternative] != list(range(1, len(alternative) + 1)):
                        continue
                    if events and len(alternative) != len(events):
                        continue
                    if len({v for _, v, _ in alternative}) != 1:
                        continue
                    if any(a[2] >= b[2] for a, b in zip(alternative, alternative[1:])):
                        continue
                    sequences.append([{"video_id": video, "frame_idx": frame} for _, video, frame in alternative])
                targets = [t for seq in sequences for t in seq]
            elif kind == "QA":
                targets = [{"video_id": video, "frame_idx": frame, "answers": [answer]} for video, frame, answer in q.gt if answer.strip()]
            else:
                targets = [{"video_id": video, "frame_idx": frame} for video, frame in q.gt]
            profiles = {profile(t["video_id"]) for t in targets}
            if not targets or len(profiles) != 1:
                skipped.append({"query_id": q.query_id, "type": kind, "reason": "missing ground truth/answer or mixed corpora"})
                continue
            qid = f"{kind}:{q.query_id}"
            # Group repeated questions across task types by normalized text to
            # keep duplicates on the same side of the development/test split.
            text_key = " ".join(unicodedata.normalize("NFC", q.text).casefold().split())
            split = "dev" if int(hashlib.sha256(text_key.encode()).hexdigest()[:8], 16) % 5 == 0 else "test"
            rows.append({"query_id": qid, "query": q.text, "query_type": kind, "split": split,
                         "retrieval_database": profiles.pop(), "image_models": ["pe"], "targets": targets,
                         **({"sequences": sequences} if sequences else {})})
    return rows, {"tkis_exclusions": reasons, "skipped": skipped}


def merge_additional(rows, additional):
    """Preserve dev exposure; promote new same-video labels to known base dev.

    Exact repeated text is merged only for the same task/corpus. Incoming dev
    questions never move to test, including exact duplicates. Existing video
    overlap across splits is reported, never silently advertised as leak-free.
    """
    def key(query):
        return " ".join(unicodedata.normalize("NFC", query["query"]).casefold().split())

    result = copy.deepcopy(rows)
    by_text = {key(q): q for q in result}
    ids = {q["query_id"] for q in result}
    base_splits = {}
    for query in rows:
        for video in {t["video_id"] for t in query["targets"]}:
            base_splits.setdefault(video, set()).add(query["split"])
    audit = {"exact_duplicates_merged": [], "split_changes": [], "video_overlap": []}
    for query in additional:
        query = copy.deepcopy(query)
        validate_hints(query)
        existing = by_text.get(key(query))
        if existing:
            if (existing["query_type"], existing["retrieval_database"]) != (query["query_type"], query["retrieval_database"]):
                raise ValueError("Identical text has conflicting task type or corpus; resolve it before merging")
            if query.get("hints_vi"):
                if existing.get("hints_vi") and existing["hints_vi"] != query["hints_vi"]:
                    raise ValueError("Repeated query has conflicting reviewed hint boundaries; audit before merging")
                existing["hints_vi"] = copy.deepcopy(query["hints_vi"])
                validate_hints(existing)
            for field in ("targets", "sequences"):
                values = list(existing.get(field, []))
                for value in query.get(field, []):
                    if value not in values:
                        values.append(value)
                if values:
                    existing[field] = values
            existing.setdefault("additional_ground_truth_sources", []).append({
                "query_id": query["query_id"], "ground_truth_source": query.get("ground_truth_source"),
                "provenance": query.get("provenance"),
            })
            audit["exact_duplicates_merged"].append({"kept": existing["query_id"], "merged": query["query_id"]})
            if query["split"] == "dev" and existing["split"] != "dev":
                # An incoming label already used in dev must not become test
                # merely because its repeated question was test in the base.
                existing["split"] = "dev"
                audit["split_changes"].append({"query_id": existing["query_id"], "from": "test", "to": "dev",
                                               "reason": "identical question already exposed in additional dev labels"})
            ids.add(query["query_id"])
            continue
        if query["query_id"] in ids:
            raise ValueError("Conflicting query_id while merging datasets")
        known = set().union(*(base_splits.get(t["video_id"], set()) for t in query["targets"]))
        if len(known) == 1:
            desired = next(iter(known))
            if desired == "dev" and query["split"] != "dev":
                change = {"query_id": query["query_id"], "from": query["split"], "to": desired,
                          "reason": "same target video as a query in the frozen base dataset"}
                query["split"] = desired
                query["split_assignment"] = change
                audit["split_changes"].append(change)
        elif len(known) > 1:
            raise ValueError("New query overlaps base dev AND test videos; resolve base splits before merging")
        result.append(query)
        ids.add(query["query_id"])
        by_text[key(query)] = query
    by_video = {}
    for query in result:
        for video in {t["video_id"] for t in query["targets"]}:
            by_video.setdefault(video, []).append({"query_id": query["query_id"], "split": query["split"]})
    audit["video_overlap"] = [{"video_id": video, "queries": queries,
                               "cross_split": len({q["split"] for q in queries}) > 1}
                              for video, queries in sorted(by_video.items()) if len(queries) > 1]
    return result, audit


def group_video_splits(rows):
    """Keep all labels for connected target videos on one side of the split.

    Dev wins if a component was already exposed during development. This only
    moves test questions to dev; previously used dev questions never become test.
    """
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for i, query in enumerate(rows):
        for video in {t["video_id"] for t in query["targets"]}:
            if video in owners:
                parent[find(i)] = find(owners[video])
            owners[video] = i
    dev_groups = {find(i) for i, query in enumerate(rows) if query["split"] == "dev"}
    result, changes = [], []
    for i, query in enumerate(rows):
        query = dict(query)
        if query["split"] != "dev" and find(i) in dev_groups:
            change = {"query_id": query["query_id"], "from": query["split"], "to": "dev",
                      "reason": "target-video component includes a development query"}
            changes.append(change)
            query["split"] = "dev"
            query["split_assignment"] = change
        result.append(query)
    return result, {"policy": "dev_dominant_target_video_components", "changes": changes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--additional-dataset", action="append", default=[],
                   help="Reviewed agent JSONL to merge with workbook labels (repeatable)")
    p.add_argument("--group-video-splits", action="store_true",
                   help="Move test queries sharing target videos with dev into dev; use a new dataset version")
    args = p.parse_args()
    rows, audit = export()
    audit["additional_datasets"] = []
    for source in args.additional_dataset:
        additional = load_queries(source)
        rows, merge_audit = merge_additional(rows, additional)
        audit["additional_datasets"].append({"path": source,
                                            "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                                            **merge_audit})
    if args.group_video_splits:
        rows, audit["split_grouping"] = group_video_splits(rows)
    video_sides = {}
    for row in rows:
        for target in row["targets"]:
            video_sides.setdefault(target["video_id"], set()).add(row["split"])
    audit["final_split_audit"] = {
        "dev_queries": sum(r["split"] == "dev" for r in rows),
        "test_queries": sum(r["split"] == "test" for r in rows),
        "cross_split_target_videos": sorted(v for v, sides in video_sides.items() if len(sides) > 1),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("Output exists; choose a new dataset path")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    load_queries(path)
    path.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Exported {len(rows)} labeled queries; dev={sum(r['split'] == 'dev' for r in rows)}; test={sum(r['split'] == 'test' for r in rows)}")


if __name__ == "__main__":
    main()
