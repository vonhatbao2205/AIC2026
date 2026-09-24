#!/usr/bin/env python3
"""Build ``backend/app/traffic_cameras.json``: where and when each traffic-camera video was filmed.

The N001–N100 videos are fixed street cameras whose banner prints the junction,
the date and the clock on every frame. OCR batch 2 split that banner into
``banner_camera``, ``banner_date`` and ``clock`` (AIC2026_Batch2_OCR_HANDOFF.md
§7.4, §10.2), and this script turns those per-frame readings into the catalogue
the backend matches a query against (``app/traffic.py``):

* **camera** — the junction, one per video. The banner spells it a dozen ways
  (``NgThiMinhKhai-DinhTienHoang 1``, ``NKKN - Ly Chinh Thang``, OCR variants), so
  each video takes the majority of its frames' labels folded to a compact key, and
  ``CAMERAS`` maps that key to the streets in proper Vietnamese. A key missing from
  ``CAMERAS`` stops the build rather than shipping a camera no query can reach.
  The N folder is not the camera: N016 holds five different junctions.
* **dates** — the banner date, per video.
* **minutes** — ``[minute_of_day, first_n, last_n]`` runs covering the whole video.
  The banner clock minus the frame's ``pts_time`` is the moment the recording
  started, and it holds to 0.4 s (p90) in every one of the 298 videos, so each
  video's start is the median of those differences and every keyframe's wall
  clock is ``start + pts_time``. That dates the frames whose banner OCR could not
  read too (N011-V003 loses its clock after 19:01), and a misread clock is one
  outlier against a median rather than a window stretched across the video.

Only reads Elastic. Run it again only if the OCR or keyframe snapshot changes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elastic_upload import DEFAULT_INFOSHOTPP_MAP_INDEX, ElasticClient, read_secret

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from app.text_normalization import fold_vietnamese  # noqa: E402

DEFAULT_OCR_INDEX = "aic26_ocr_keyframes_v2"
DEFAULT_OUT = Path("backend/app/traffic_cameras.json")
EXPECTED_VIDEOS = 298
#: A date read on fewer frames than this share of a video is an OCR misread.
DATE_MIN_SHARE = 0.2
_CLOCK = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

#: Camera key (the banner label folded to [a-z0-9]) -> (display label, streets).
#: Streets are what a query names; landmarks count as streets. "1"/"2" suffixes are
#: two cameras on the same junction and keep separate keys.
CAMERAS: dict[str, tuple[str, tuple[str, ...]]] = {
    "anduongvuonglehongphong": ("An Dương Vương – Lê Hồng Phong", ("An Dương Vương", "Lê Hồng Phong")),
    "anduongvuongnguyenvancu": ("An Dương Vương – Nguyễn Văn Cừ", ("An Dương Vương", "Nguyễn Văn Cừ")),
    "anduongvuongtranphu": ("An Dương Vương – Trần Phú", ("An Dương Vương", "Trần Phú")),
    "cachmangthang8nguyendinhchieu": ("Cách Mạng Tháng 8 – Nguyễn Đình Chiểu", ("Cách Mạng Tháng 8", "Nguyễn Đình Chiểu")),
    "cachmangthangtamrachbungbinh": ("Cách Mạng Tháng 8 – Rạch Bùng Binh", ("Cách Mạng Tháng 8", "Rạch Bùng Binh")),
    "caothangvovantan1": ("Cao Thắng – Võ Văn Tần (1)", ("Cao Thắng", "Võ Văn Tần")),
    "caothangvovantan2": ("Cao Thắng – Võ Văn Tần (2)", ("Cao Thắng", "Võ Văn Tần")),
    "cmt8buithixuan": ("Cách Mạng Tháng 8 – Bùi Thị Xuân", ("Cách Mạng Tháng 8", "Bùi Thị Xuân")),
    "cmt8vovantan": ("Cách Mạng Tháng 8 – Võ Văn Tần", ("Cách Mạng Tháng 8", "Võ Văn Tần")),
    "congquynhbuivien": ("Cống Quỳnh – Bùi Viện", ("Cống Quỳnh", "Bùi Viện")),
    "congquynhphamvietchanh": ("Cống Quỳnh – Phạm Viết Chánh", ("Cống Quỳnh", "Phạm Viết Chánh")),
    "congtruongdanchu1": ("Công trường Dân Chủ (1)", ("Công trường Dân Chủ",)),
    "congtruongdanchu2": ("Công trường Dân Chủ (2)", ("Công trường Dân Chủ",)),
    "congtruongmelinh": ("Công trường Mê Linh", ("Công trường Mê Linh",)),
    "congviendacaucalmette": ("Công viên – cầu Calmette", ("Calmette",)),
    "dbphutruongdinh": ("Điện Biên Phủ – Trương Định", ("Điện Biên Phủ", "Trương Định")),
    "haibatrunglychinhthang": ("Hai Bà Trưng – Lý Chính Thắng", ("Hai Bà Trưng", "Lý Chính Thắng")),
    "haibatrungnguyendinhchieu": ("Hai Bà Trưng – Nguyễn Đình Chiểu", ("Hai Bà Trưng", "Nguyễn Đình Chiểu")),
    "haibatrungtrancaovan": ("Hai Bà Trưng – Trần Cao Vân", ("Hai Bà Trưng", "Trần Cao Vân")),
    "haibatrungtranquoctoan": ("Hai Bà Trưng – Trần Quốc Toản", ("Hai Bà Trưng", "Trần Quốc Toản")),
    "haithuonglanongchauvanliem": ("Hải Thượng Lãn Ông – Châu Văn Liêm", ("Hải Thượng Lãn Ông", "Châu Văn Liêm")),
    "haugiangminhphung": ("Hậu Giang – Minh Phụng", ("Hậu Giang", "Minh Phụng")),
    "haugiangnguyenvanluong": ("Hậu Giang – Nguyễn Văn Luông", ("Hậu Giang", "Nguyễn Văn Luông")),
    "hongbangngoquyen": ("Hồng Bàng – Ngô Quyền (1)", ("Hồng Bàng", "Ngô Quyền")),
    "hongbangngoquyen2": ("Hồng Bàng – Ngô Quyền (2)", ("Hồng Bàng", "Ngô Quyền")),
    "hongbangnguyenthinho": ("Hồng Bàng – Nguyễn Thị Nhỏ", ("Hồng Bàng", "Nguyễn Thị Nhỏ")),
    "hongbangphudongthienvuong": ("Hồng Bàng – Phù Đổng Thiên Vương", ("Hồng Bàng", "Phù Đổng Thiên Vương")),
    "hongbangtauyen": ("Hồng Bàng – Tạ Uyên", ("Hồng Bàng", "Tạ Uyên")),
    "hungvuonglehongphong": ("Hùng Vương – Lê Hồng Phong", ("Hùng Vương", "Lê Hồng Phong")),
    "hungvuongngogiatu": ("Hùng Vương – Ngô Gia Tự", ("Hùng Vương", "Ngô Gia Tự")),
    "hungvuongnguyenchithanh1": ("Hùng Vương – Nguyễn Chí Thanh", ("Hùng Vương", "Nguyễn Chí Thanh")),
    "huynhtanphatcautanthuan1": ("Huỳnh Tấn Phát – cầu Tân Thuận", ("Huỳnh Tấn Phát", "Tân Thuận")),
    "huynhtanphathoangquocviet": ("Huỳnh Tấn Phát – Hoàng Quốc Việt", ("Huỳnh Tấn Phát", "Hoàng Quốc Việt")),
    "huynhtanphatphuthuan": ("Huỳnh Tấn Phát – Phú Thuận", ("Huỳnh Tấn Phát", "Phú Thuận")),
    "huynhtanphattrantrongcung": ("Huỳnh Tấn Phát – Trần Trọng Cung", ("Huỳnh Tấn Phát", "Trần Trọng Cung")),
    "khanhhoicaukenhte": ("Khánh Hội – cầu Kênh Tẻ", ("Khánh Hội", "Kênh Tẻ")),
    "khanhhoiduongso41": ("Khánh Hội – Đường số 41", ("Khánh Hội", "Đường số 41")),
    "khanhhoivinhhoi": ("Khánh Hội – Vĩnh Hội", ("Khánh Hội", "Vĩnh Hội")),
    "kydongbahuyenthanhquan": ("Kỳ Đồng – Bà Huyện Thanh Quan", ("Kỳ Đồng", "Bà Huyện Thanh Quan")),
    "lcthangnguyenthong": ("Lý Chính Thắng – Nguyễn Thông", ("Lý Chính Thắng", "Nguyễn Thông")),
    "lcthangtruongdinh": ("Lý Chính Thắng – Trương Định", ("Lý Chính Thắng", "Trương Định")),
    "leduanmacdinhchi": ("Lê Duẩn – Mạc Đĩnh Chi", ("Lê Duẩn", "Mạc Đĩnh Chi")),
    "leduannguyenbinhkhiem": ("Lê Duẩn – Nguyễn Bỉnh Khiêm", ("Lê Duẩn", "Nguyễn Bỉnh Khiêm")),
    "leduanphamngocthach": ("Lê Duẩn – Phạm Ngọc Thạch", ("Lê Duẩn", "Phạm Ngọc Thạch")),
    "lehongphongtranphu": ("Lê Hồng Phong – Trần Phú", ("Lê Hồng Phong", "Trần Phú")),
    "lelaiphamhongthai": ("Lê Lai – Phạm Hồng Thái", ("Lê Lai", "Phạm Hồng Thái")),
    "lelaiphanchutrinh": ("Lê Lai – Phan Chu Trinh", ("Lê Lai", "Phan Chu Trinh")),
    "leloipasteur": ("Lê Lợi – Pasteur", ("Lê Lợi", "Pasteur")),
    "lethanhtondongkhoi": ("Lê Thánh Tôn – Đồng Khởi", ("Lê Thánh Tôn", "Đồng Khởi")),
    "levansytranquangdieu": ("Lê Văn Sỹ – Trần Quang Diệu", ("Lê Văn Sỹ", "Trần Quang Diệu")),
    "levansytruongsa": ("Lê Văn Sỹ – Trường Sa", ("Lê Văn Sỹ", "Trường Sa")),
    "lychinhthangtranquocthao": ("Lý Chính Thắng – Trần Quốc Thảo", ("Lý Chính Thắng", "Trần Quốc Thảo")),
    "lythuongkietnguyenchithanh": ("Lý Thường Kiệt – Nguyễn Chí Thanh", ("Lý Thường Kiệt", "Nguyễn Chí Thanh")),
    "namkykhoinghiahamnghi": ("Nam Kỳ Khởi Nghĩa – Hàm Nghi", ("Nam Kỳ Khởi Nghĩa", "Hàm Nghi")),
    "namkykhoinghialytutrong": ("Nam Kỳ Khởi Nghĩa – Lý Tự Trọng", ("Nam Kỳ Khởi Nghĩa", "Lý Tự Trọng")),
    "namkykhoinghiavovantan": ("Nam Kỳ Khởi Nghĩa – Võ Văn Tần", ("Nam Kỳ Khởi Nghĩa", "Võ Văn Tần")),
    "nbknguyendinhchieu": ("Nguyễn Bỉnh Khiêm – Nguyễn Đình Chiểu", ("Nguyễn Bỉnh Khiêm", "Nguyễn Đình Chiểu")),
    "nga6conghoa2": ("Ngã sáu Cộng Hòa (2)", ("Ngã sáu", "Cộng Hòa")),
    "nga6nguyentriphuong2": ("Ngã sáu Nguyễn Tri Phương (2)", ("Ngã sáu", "Nguyễn Tri Phương")),
    "ngthiminhkhaidinhtienhoang1": ("Nguyễn Thị Minh Khai – Đinh Tiên Hoàng (1)", ("Nguyễn Thị Minh Khai", "Đinh Tiên Hoàng")),
    "ngthiminhkhaidinhtienhoang2": ("Nguyễn Thị Minh Khai – Đinh Tiên Hoàng (2)", ("Nguyễn Thị Minh Khai", "Đinh Tiên Hoàng")),
    "ngthiminhkhaingbinhkhiem": ("Nguyễn Thị Minh Khai – Nguyễn Bỉnh Khiêm", ("Nguyễn Thị Minh Khai", "Nguyễn Bỉnh Khiêm")),
    "nguyenchithanhngoquyen": ("Nguyễn Chí Thanh – Ngô Quyền", ("Nguyễn Chí Thanh", "Ngô Quyền")),
    "nguyenchithanhnguyenkim": ("Nguyễn Chí Thanh – Nguyễn Kim", ("Nguyễn Chí Thanh", "Nguyễn Kim")),
    "nguyenchithanhthuankieu": ("Nguyễn Chí Thanh – Thuận Kiều", ("Nguyễn Chí Thanh", "Thuận Kiều")),
    "nguyencongtrucaucalmett2": ("Nguyễn Công Trứ – cầu Calmette", ("Nguyễn Công Trứ", "Calmette")),
    "nguyendinhchieucaothang": ("Nguyễn Đình Chiểu – Cao Thắng", ("Nguyễn Đình Chiểu", "Cao Thắng")),
    "nguyendinhchieutruongdinh": ("Nguyễn Đình Chiểu – Trương Định", ("Nguyễn Đình Chiểu", "Trương Định")),
    "nguyenducongxaparis": ("Nguyễn Du – Công xã Paris", ("Nguyễn Du", "Công xã Paris")),
    "nguyenhuucanhtonducthang": ("Nguyễn Hữu Cảnh – Tôn Đức Thắng", ("Nguyễn Hữu Cảnh", "Tôn Đức Thắng")),
    "nguyenluongbanghoangquocviet": ("Nguyễn Lương Bằng – Hoàng Quốc Việt", ("Nguyễn Lương Bằng", "Hoàng Quốc Việt")),
    "nguyentatthanhcautanthuan1": ("Nguyễn Tất Thành – cầu Tân Thuận", ("Nguyễn Tất Thành", "Tân Thuận")),
    "nguyentatthanhtondan": ("Nguyễn Tất Thành – Tôn Đản", ("Nguyễn Tất Thành", "Tôn Đản")),
    "nguyenthaihocphamngulao": ("Nguyễn Thái Học – Phạm Ngũ Lão", ("Nguyễn Thái Học", "Phạm Ngũ Lão")),
    "nguyenthiminhkhaicongquynh": ("Nguyễn Thị Minh Khai – Cống Quỳnh", ("Nguyễn Thị Minh Khai", "Cống Quỳnh")),
    "nguyenthiminhkhaiphamngocthach": ("Nguyễn Thị Minh Khai – Phạm Ngọc Thạch", ("Nguyễn Thị Minh Khai", "Phạm Ngọc Thạch")),
    "nguyenthiminhkhaitruongdinh": ("Nguyễn Thị Minh Khai – Trương Định", ("Nguyễn Thị Minh Khai", "Trương Định")),
    "nguyenthinghialelai": ("Nguyễn Thị Nghĩa – Lê Lai", ("Nguyễn Thị Nghĩa", "Lê Lai")),
    "nguyenthithaphuynhtanphat": ("Nguyễn Thị Thập – Huỳnh Tấn Phát", ("Nguyễn Thị Thập", "Huỳnh Tấn Phát")),
    "nguyenthithaptanmy": ("Nguyễn Thị Thập – Tân Mỹ", ("Nguyễn Thị Thập", "Tân Mỹ")),
    "nguyentraicongquynh1": ("Nguyễn Trãi – Cống Quỳnh (1)", ("Nguyễn Trãi", "Cống Quỳnh")),
    "nguyentraicongquynh2": ("Nguyễn Trãi – Cống Quỳnh (2)", ("Nguyễn Trãi", "Cống Quỳnh")),
    "nguyentrainguyencutrinh": ("Nguyễn Trãi – Nguyễn Cư Trinh", ("Nguyễn Trãi", "Nguyễn Cư Trinh")),
    "nguyentraitranphu": ("Nguyễn Trãi – Trần Phú", ("Nguyễn Trãi", "Trần Phú")),
    "nguyenvancunguyentrai": ("Nguyễn Văn Cừ – Nguyễn Trãi", ("Nguyễn Văn Cừ", "Nguyễn Trãi")),
    "nguyenvancutranhungdao2": ("Nguyễn Văn Cừ – Trần Hưng Đạo (2)", ("Nguyễn Văn Cừ", "Trần Hưng Đạo")),
    "nguyenvancutranhungdao3": ("Nguyễn Văn Cừ – Trần Hưng Đạo (3)", ("Nguyễn Văn Cừ", "Trần Hưng Đạo")),
    "nguyenvanlinhcautanthuan2": ("Nguyễn Văn Linh – cầu Tân Thuận", ("Nguyễn Văn Linh", "Tân Thuận")),
    "nguyenvanthutrandoankhanh": ("Nguyễn Văn Thủ – Trần Doãn Khanh", ("Nguyễn Văn Thủ", "Trần Doãn Khanh")),
    "nhcanhnbkhiemcctv43": ("Nguyễn Hữu Cảnh – Nguyễn Bỉnh Khiêm", ("Nguyễn Hữu Cảnh", "Nguyễn Bỉnh Khiêm")),
    "nkknlychinhthang": ("Nam Kỳ Khởi Nghĩa – Lý Chính Thắng", ("Nam Kỳ Khởi Nghĩa", "Lý Chính Thắng")),
    "nkknnguyendu": ("Nam Kỳ Khởi Nghĩa – Nguyễn Du", ("Nam Kỳ Khởi Nghĩa", "Nguyễn Du")),
    "nkkntranquoctoan": ("Nam Kỳ Khởi Nghĩa – Trần Quốc Toản", ("Nam Kỳ Khởi Nghĩa", "Trần Quốc Toản")),
    "ntmknguyenthuonghien": ("Nguyễn Thị Minh Khai – Nguyễn Thượng Hiền", ("Nguyễn Thị Minh Khai", "Nguyễn Thượng Hiền")),
    "ntmktonthattung": ("Nguyễn Thị Minh Khai – Tôn Thất Tùng", ("Nguyễn Thị Minh Khai", "Tôn Thất Tùng")),
    "nutgiaonga6nguyentriphuong1": ("Ngã sáu Nguyễn Tri Phương (1)", ("Ngã sáu", "Nguyễn Tri Phương")),
    "nutgiaongasauconghoa1": ("Ngã sáu Cộng Hòa (1)", ("Ngã sáu", "Cộng Hòa")),
    "phamhuulauhuynhtanphat": ("Phạm Hữu Lầu – Huỳnh Tấn Phát", ("Phạm Hữu Lầu", "Huỳnh Tấn Phát")),
}


def camera_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold_vietnamese(label))


def seconds_of_day(clock: str | None) -> int | None:
    match = _CLOCK.fullmatch((clock or "").strip())
    if not match:
        return None
    hour, minute, second = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return hour * 3600 + minute * 60 + second if hour < 24 and minute < 60 and second < 60 else None


def recording_start(rows: list[dict[str, Any]]) -> float:
    """Seconds of day at pts 0: the median of banner clock minus pts_time."""
    offsets = [
        seconds - row["pts_time"]
        for row in rows
        if (seconds := seconds_of_day(row.get("clock"))) is not None and row.get("pts_time") is not None
    ]
    if not offsets:
        raise SystemExit(f"{rows[0]['video_id']}: no readable banner clock")
    return statistics.median(offsets)


def minute_runs(rows: list[dict[str, Any]], start: float) -> list[list[int]]:
    """`[minute_of_day, first_n, last_n]` runs, one per wall-clock minute, in keyframe order."""
    runs: list[list[int]] = []
    for row in sorted(rows, key=lambda item: item["keyframe_n"]):
        minute = int((start + row["pts_time"]) // 60) % 1440
        if runs and runs[-1][0] == minute:
            runs[-1][2] = row["keyframe_n"]
        else:
            runs.append([minute, row["keyframe_n"], row["keyframe_n"]])
    return runs


def clock_text(seconds: float) -> str:
    seconds = int(round(seconds)) % 86400
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def fetch_n_frames(client: ElasticClient, index: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    search_after = None
    while True:
        body: dict[str, Any] = {
            "size": 10_000,
            "query": {"term": {"profile": "N"}},
            "_source": ["video_id", "keyframe_n", "pts_time", "banner_camera", "banner_date", "clock"],
            "sort": [{"video_id": "asc"}, {"keyframe_n": "asc"}],
        }
        if search_after:
            body["search_after"] = search_after
        hits = client.json_request("POST", f"{index}/_search", payload=body, timeout=120)["hits"]["hits"]
        frames.extend(hit["_source"] for hit in hits)
        if len(hits) < body["size"]:
            return frames
        search_after = hits[-1]["sort"]


def keyframe_counts(client: ElasticClient, index: str) -> dict[str, int]:
    body = {
        "size": 0,
        "query": {"prefix": {"video_id": "N"}},
        "aggs": {"videos": {"terms": {"field": "video_id", "size": 1000}, "aggs": {"n": {"max": {"field": "keyframe_n"}}}}},
    }
    buckets = client.json_request("POST", f"{index}/_search", payload=body, timeout=120)["aggregations"]["videos"]["buckets"]
    return {bucket["key"]: int(bucket["n"]["value"]) for bucket in buckets}


def build_catalog(frames: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_video[frame["video_id"]].append(frame)
    if set(by_video) != set(counts) or len(counts) != EXPECTED_VIDEOS:
        raise SystemExit(f"{len(by_video)} videos with OCR, {len(counts)} in the keyframe map, expected {EXPECTED_VIDEOS}")

    videos: dict[str, dict[str, Any]] = {}
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    unknown: dict[str, list[str]] = {}
    for video_id in sorted(by_video):
        rows = by_video[video_id]
        votes: Counter[str] = Counter()
        for row in rows:
            if row.get("banner_camera"):
                votes[camera_key(row["banner_camera"])] += 1
        key = votes.most_common(1)[0][0]
        for row in rows:
            if row.get("banner_camera") and camera_key(row["banner_camera"]) == key:
                labels[key][row["banner_camera"]] += 1
        if key not in CAMERAS:
            unknown.setdefault(key, []).append(video_id)
        # Every N keyframe was OCR'd (none is black), so the rows ARE the video.
        if sorted(row["keyframe_n"] for row in rows) != list(range(1, counts[video_id] + 1)):
            raise SystemExit(f"{video_id}: OCR rows do not cover keyframes 1..{counts[video_id]}")
        dates = Counter(row["banner_date"] for row in rows if row.get("banner_date"))
        floor = max(3, DATE_MIN_SHARE * len(rows))
        start = recording_start(rows)
        last_pts = max(row["pts_time"] for row in rows)
        videos[video_id] = {
            "camera": key,
            "keyframes": counts[video_id],
            "dates": sorted(date for date, count in dates.items() if count >= floor),
            "clock": [clock_text(start), clock_text(start + last_pts)],
            "minutes": minute_runs(rows, start),
        }
    if unknown:
        raise SystemExit(
            "Cameras missing from CAMERAS (add their streets): "
            + "; ".join(f"{key} {labels[key].most_common(1)[0][0]!r} {videos}" for key, videos in sorted(unknown.items()))
        )

    cameras = []
    for key, (label, streets) in sorted(CAMERAS.items()):
        members = sorted(video for video, entry in videos.items() if entry["camera"] == key)
        if not members:
            raise SystemExit(f"CAMERAS lists {key}, which no video shows")
        cameras.append(
            {"id": key, "label": label, "banner": labels[key].most_common(1)[0][0], "streets": list(streets), "videos": members}
        )
    return {"cameras": cameras, "videos": videos}


def render(document: dict[str, Any]) -> str:
    """JSON with one camera and one video per line, so a rebuild reads as a diff."""
    def compact(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    head = {key: value for key, value in document.items() if key not in {"cameras", "videos"}}
    lines = ["{"]
    lines += [f"  {compact(key)}: {compact(value)}," for key, value in head.items()]
    lines.append('  "cameras": [')
    lines.append(",\n".join(f"    {compact(camera)}" for camera in document["cameras"]))
    lines.append("  ],")
    lines.append('  "videos": {')
    lines.append(",\n".join(f"    {compact(key)}: {compact(value)}" for key, value in document["videos"].items()))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-index", default=DEFAULT_OCR_INDEX)
    parser.add_argument("--map-index", default=DEFAULT_INFOSHOTPP_MAP_INDEX)
    parser.add_argument("--endpoint-file", type=Path, default=Path("API_KEY/elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("API_KEY/elastic_apikey.txt"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    client = ElasticClient(read_secret(args.endpoint_file), read_secret(args.api_key_file))
    frames = fetch_n_frames(client, args.ocr_index)
    catalog = build_catalog(frames, keyframe_counts(client, args.map_index))
    document = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"ocr_index": args.ocr_index, "map_index": args.map_index, "frames": len(frames)},
        **catalog,
    }
    args.out.write_text(render(document), encoding="utf-8")
    runs = sum(len(video["minutes"]) for video in catalog["videos"].values())
    print(f"{args.out}: {len(catalog['cameras'])} cameras, {len(catalog['videos'])} videos, {runs} minute runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
