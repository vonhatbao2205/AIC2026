"""Đọc ground truth từ ba workbook TKIS / TRAKE / QA.

Mục tiêu: biến ba file Excel do nhóm ghi tay thành cấu trúc Python thống nhất,
đồng thời ghi lại lý do loại bỏ từng query để bảng thống kê trong báo cáo có thể
truy vết được. Không suy đoán: query nào thiếu video/frame thì bị loại và ghi rõ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent


def _s(value) -> str:
    return "" if value is None else str(value).strip()


def _frames(cell: str) -> list[int]:
    """Tách '25815, 25605' hoặc '15960 → 16020' thành danh sách frame index."""
    return [int(tok) for tok in re.findall(r"\d+", cell)]


@dataclass
class TkisQuery:
    query_id: str
    phase: str
    group: str
    text: str
    gt: list[tuple[str, int]] = field(default_factory=list)  # (video_id, frame_idx)
    dbc: str = ""
    status: str = ""
    recall_note: str = ""

    @property
    def gt_videos(self) -> set[str]:
        return {v for v, _ in self.gt}


@dataclass
class TrakeQuery:
    query_id: str
    phase: str
    text: str
    events: list[str] = field(default_factory=list)
    # alternative -> danh sách (thứ tự event, video_id, frame_idx)
    alternatives: dict[str, list[tuple[int, str, int]]] = field(default_factory=dict)
    status: str = ""

    @property
    def n_events(self) -> int:
        return max((len(v) for v in self.alternatives.values()), default=len(self.events))


@dataclass
class QaQuery:
    query_id: str
    phase: str
    text: str
    gt: list[tuple[str, int, str]] = field(default_factory=list)  # (video, frame, answer)
    status: str = ""

    @property
    def gt_videos(self) -> set[str]:
        return {v for v, _, _ in self.gt}


def load_tkis(path: Path | None = None) -> tuple[list[TkisQuery], dict[str, int]]:
    path = path or ROOT / "TKIS_queries.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    # Sheet chi tiết là nguồn ground truth đáng tin nhất: mỗi dòng đúng một frame.
    detail: dict[str, list[tuple[str, int]]] = {}
    ws = wb["GT từng frame"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:
            continue
        qid, video, frame = _s(row[1]), _s(row[3]), _s(row[4])
        if not qid or not video or not frame:
            continue
        for fr in _frames(frame):
            detail.setdefault(qid, []).append((video, fr))

    queries: list[TkisQuery] = []
    reasons = {"no_gt": 0, "gt_zero": 0, "missing_frame": 0, "ok": 0}
    ws = wb["TKIS tổng hợp"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5:
            continue
        qid, text = _s(row[2]), _s(row[4])
        if not qid or not text:
            continue
        q = TkisQuery(
            query_id=qid,
            phase=_s(row[1]),
            group=_s(row[3]),
            text=text,
            dbc=_s(row[9]),
            status=_s(row[10]),
            recall_note=_s(row[11]),
        )
        # Ưu tiên sheet chi tiết; nếu không có thì đọc cột GT nguyên bản.
        q.gt = sorted(set(detail.get(qid, [])))
        if not q.gt:
            video, frame = _s(row[6]), _s(row[7])
            if video and frame:
                q.gt = sorted({(video, fr) for fr in _frames(frame)})

        if not q.gt:
            reasons["no_gt"] += 1
            continue
        if _s(row[8]) == "0":
            reasons["gt_zero"] += 1
            continue
        reasons["ok"] += 1
        queries.append(q)

    return queries, reasons


def load_trake(path: Path | None = None) -> list[TrakeQuery]:
    path = path or ROOT / "TRAKE_queries.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    detail: dict[str, dict[str, list[tuple[int, str, int]]]] = {}
    ws = wb["TRAKE event chi tiết"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:
            continue
        qid, alt, order = _s(row[3]), _s(row[4]) or "1", _s(row[5])
        video, frame = _s(row[8]), _s(row[9])
        if not qid or not video or not frame or not order:
            continue
        fr = _frames(frame)
        if not fr:
            continue
        detail.setdefault(qid, {}).setdefault(alt, []).append((int(order), video, fr[0]))

    queries: list[TrakeQuery] = []
    ws = wb["TRAKE tổng hợp"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5:
            continue
        qid, text = _s(row[4]), _s(row[5])
        if not qid or not text:
            continue
        events = [e.strip() for e in _s(row[7]).split("\n") if e.strip()]
        alts = {k: sorted(v) for k, v in detail.get(qid, {}).items()}
        queries.append(
            TrakeQuery(
                query_id=qid,
                phase=_s(row[2]),
                text=text,
                events=events,
                alternatives=alts,
                status=_s(row[15]),
            )
        )
    return queries


def load_qa(path: Path | None = None) -> list[QaQuery]:
    path = path or ROOT / "QA_queries.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    detail: dict[str, list[tuple[str, int, str]]] = {}
    ws = wb["QA GT chi tiết"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:
            continue
        qid, video, frame, ans = _s(row[3]), _s(row[4]), _s(row[5]), _s(row[6])
        if not qid or not video or not frame:
            continue
        for fr in _frames(frame):
            detail.setdefault(qid, []).append((video, fr, ans))

    queries: list[QaQuery] = []
    ws = wb["QA tổng hợp"]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5:
            continue
        qid, text = _s(row[4]), _s(row[5])
        if not qid or not text:
            continue
        queries.append(
            QaQuery(
                query_id=qid,
                phase=_s(row[2]),
                text=text,
                gt=sorted(set(detail.get(qid, []))),
                status=_s(row[14]),
            )
        )
    return queries


if __name__ == "__main__":
    tkis, reasons = load_tkis()
    print(f"T-KIS: {len(tkis)} query có GT dùng được; lý do loại: {reasons}")
    print(f"  tổng frame GT: {sum(len(q.gt) for q in tkis)}")

    trake = load_trake()
    with_gt = [q for q in trake if q.alternatives]
    print(f"TRAKE: {len(trake)} query, {len(with_gt)} có GT")
    for q in trake:
        print(f"  {q.query_id}: {q.n_events} event, {len(q.alternatives)} phương án GT")

    qa = load_qa()
    with_gt = [q for q in qa if q.gt]
    print(f"QA: {len(qa)} query, {len(with_gt)} có GT, {sum(len(q.gt) for q in qa)} bộ (video,frame,answer)")
