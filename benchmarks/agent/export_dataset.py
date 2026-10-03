"""Export existing workbook labels without inventing missing ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from benchmarks.gt_loader import load_qa, load_tkis, load_trake
from backend.app.trake_events import split_marked_events


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows, audit = export()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("Output exists; choose a new dataset path")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    path.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Exported {len(rows)} labeled queries; dev={sum(r['split'] == 'dev' for r in rows)}; test={sum(r['split'] == 'test' for r in rows)}")


if __name__ == "__main__":
    main()
