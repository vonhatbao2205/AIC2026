"""Fetch local review clips and timestamped frames, without calling models or DRES."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from backend.app.config import get_settings
from backend.app.media import MediaUrlBuilder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def run(command, timeout=180):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-600:])
    return result.stdout


def capture(query):
    out = HERE / "evidence" / f"row-{query['workbook_rows'][0]:02d}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "inspection.json"
    if manifest.exists():
        return query["workbook_rows"][0], "cached"
    settings = get_settings().for_retrieval_database("infoshotpp")
    media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)
    candidates = [s for s in query["submissions"] if s["id"] in query["candidate_submission_ids"]]
    items = []
    probes = {}
    for submission in candidates:
        payload = submission["payload"]
        video = payload["video_id"]
        url = media.video_url(video)
        if video not in probes:
            probes[video] = json.loads(run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,avg_frame_rate,r_frame_rate,time_base,start_time",
                "-of", "json", url,
            ], 90))
        duration = float(probes[video]["format"]["duration"])
        events = payload.get("events") or [payload]
        for event in events:
            time = event.get("pts_time") if payload.get("events") else event.get("timestamp")
            if time is None:
                if not payload.get("fps") or event.get("frame_idx") is None:
                    raise ValueError("No timestamp/FPS available")
                time = event["frame_idx"] / payload["fps"]
            event_index = event.get("event_index", 0)
            stem = f"{submission['id'][:8]}-e{event_index}"
            start, end = max(0, time - 4), min(duration, time + 4)
            clip = out / f"{stem}.mp4"
            if not clip.exists():
                run([
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start), "-i", url, "-t", str(end - start),
                    "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=640:-2",
                    "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-crf", "24",
                    "-c:a", "aac", "-b:a", "64k", "-y", str(clip),
                ])
            samples = []
            for offset in (-3, -1, 0, 1, 3):
                timestamp = min(end - .04, max(start, time + offset))
                image = out / f"{stem}-{offset:+d}.jpg"
                if not image.exists():
                    run([
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
                        str(timestamp - start), "-i", str(clip), "-frames:v", "1", "-q:v", "2",
                        "-threads", "1", "-y", str(image),
                    ], 30)
                samples.append({"time_s": timestamp, "offset_s": offset, "path": str(image.relative_to(ROOT))})
            items.append({"submission_id": submission["id"], "video_id": video,
                          "event_index": event_index, "submitted_time_s": time,
                          "frame_idx": event.get("frame_idx"), "fps": payload.get("fps"),
                          "verdict": submission["verdict"], "answer": payload.get("answer"),
                          "clip_path": str(clip.relative_to(ROOT)), "clip_start_s": start,
                          "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(), "samples": samples})
    # One row per submitted moment; center column is the submitted timestamp.
    sheet = Image.new("RGB", (1600, 205 * len(items)), (20, 25, 35))
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(items):
        for column, sample in enumerate(item["samples"]):
            im = Image.open(ROOT / sample["path"])
            im.thumbnail((320, 180))
            x, y = column * 320, row * 205
            sheet.paste(im, (x, y))
            draw.text((x + 2, y + 179), f"{item['video_id']} E{item['event_index']} {sample['time_s']:.3f}s", fill="white")
            draw.text((x + 2, y + 192), f"{item['submission_id'][:8]} {item['verdict']} {item['answer'] or ''}", fill="white")
    sheet.save(out / "overview.jpg", quality=93)
    manifest.write_text(json.dumps({"query_id": query["query_id"], "probes": probes, "moments": items,
                                    "method": "Decoded +/-4 second video clips; sampled at submitted timestamp and +/-1,+/-3 seconds. Brief visual review, not continuous playback or exhaustive boundary annotation."}, ensure_ascii=False, indent=2))
    return query["workbook_rows"][0], "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="*", type=int)
    args = parser.parse_args()
    queue = json.loads((HERE / "review_queue.json").read_text())
    selected = [q for q in queue if not args.rows or q["workbook_rows"][0] in args.rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(capture, q): q for q in selected}
        for job in concurrent.futures.as_completed(jobs):
            try:
                print(job.result(), flush=True)
            except Exception as error:
                print(jobs[job]["query_id"], type(error).__name__, str(error), flush=True)


if __name__ == "__main__":
    main()
