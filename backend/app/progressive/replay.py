"""Read-only replay of committed UI snapshots; never calls a retriever or DRES.

This is event replay for debugging, not a benchmark or an algorithm ablation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def replay_events(events: list[dict]) -> list[dict]:
    session_id = None
    config = None
    issued = committed = 0
    snapshots = []
    for event in events:
        kind = event["event"]
        if kind == "created":
            if session_id is not None:
                raise ValueError("A journal must contain exactly one session")
            session_id, config = event["session_id"], event["config"]
        if session_id is None or event["session_id"] != session_id:
            raise ValueError("Missing creation event or mixed session IDs")
        if kind == "update":
            if event["revision"] != issued + 1:
                raise ValueError("Nonsequential revision in journal")
            issued = event["revision"]
        elif kind == "committed":
            revision = event["revision"]
            if revision != issued or revision <= committed:
                raise ValueError("Stale or duplicate commit in journal")
            committed = revision
            snapshots.append({**copy.deepcopy(event["snapshot"]), "session_id": session_id,
                "revision": revision, "committed_revision": revision, "pending_revision": None,
                "last_error": None, "config": copy.deepcopy(config)})
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    for snapshot in replay_events(events):
        print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
