#!/usr/bin/env python3
"""Build ``backend/app/batch2_video_guide.json``: what each S01 race recording shows.

The source is the operator's batch-2 content sheet
(``Batch2/batch2_content/batch2_ai_index.jsonl``, built by
``Batch2/batch2_content/build_batch2_sheet.py``), whose ``description`` field
summarises each video from its sampled frames. The agents' ``list_videos`` /
``folder_frames`` / ``video_outline`` tools show it next to the race stage, so an
agent can tell the opening ceremony from the finish without opening a
multi-hour recording.

Only the S01 descriptions are shipped. The N descriptions repeat the junction,
date and clock that ``traffic_cameras.json`` already holds minute by minute, and
the M descriptions are Tesseract OCR of eight frames including the news ticker,
which the Elastic OCR index and the speech transcripts cover more cleanly.

The sheet directory is not in git, so its output is copied here, next to the
code that reads it. Run this again only if the sheet is rebuilt.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_SOURCE = Path("Batch2/batch2_content/batch2_ai_index.jsonl")
DEFAULT_OUT = Path("backend/app/batch2_video_guide.json")
SERIES = ("S",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    videos: dict[str, str] = {}
    for line in args.source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        description = " ".join(str(row.get("description") or "").split())
        if row.get("group", "")[:1] in SERIES and description:
            videos[row["video_id"]] = description
    if not videos:
        raise SystemExit(f"No {'/'.join(SERIES)} descriptions in {args.source}")

    document = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(args.source),
        "videos": dict(sorted(videos.items())),
    }
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(videos)} descriptions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
