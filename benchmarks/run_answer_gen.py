"""Chấm và dò tham số cho bộ sinh 100 đáp án, hoàn toàn offline trên file cache.

Điểm chấm bám đúng thể lệ vòng sơ tuyển:

    R@k   = max R-Score trong k đáp án đầu
    Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5

Với T-KIS/QA, R-Score là 0/1 nên Final của một truy vấn chỉ phụ thuộc vị trí đáp
án đúng đầu tiên: hạng 1 → 1.0, hạng 2-5 → 0.8, 6-20 → 0.6, 21-50 → 0.4,
51-100 → 0.2, không có → 0. Với TRAKE, R-Score là tỉ lệ event khớp nên phải lấy
max thật theo từng mốc.

Ground truth của nhóm là các frame rời rạc, không phải đoạn [s, e] của BTC, nên
phải chấm với một dung sai. Mọi bảng kết quả đều in ra nhiều mức dung sai; mức
dùng để tối ưu chọn bằng --tol.

    # chuẩn hoá thang epsilon từ sai số truy xuất thực đo
    backend/.venv/bin/python benchmarks/run_answer_gen.py --mode calib

    # chấm bộ tham số mặc định
    backend/.venv/bin/python benchmarks/run_answer_gen.py --mode eval

    # dò tham số bằng coordinate descent
    backend/.venv/bin/python benchmarks/run_answer_gen.py --mode sweep --passes 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.answer_gen import (  # noqa: E402
    BANDS,
    AnswerGenParams,
    DEFAULT_PARAMS,
    build_pools,
    generate_answers,
    generate_trake_rows,
    retrieval_error_percentiles,
)

CACHE_DIR = ROOT / "benchmarks" / "cache"
RUNS_DIR = ROOT / "benchmarks" / "runs"

#: Dung sai (frame) dùng khi in bảng. 25 fps ⇒ 25 frame = 1 giây.
TOLERANCES: tuple[int, ...] = (0, 12, 25, 50, 100, 200)


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------


def final_from_rscores(rscores: Sequence[float]) -> float:
    """Final Score = trung bình của R@k tại 5 mốc, R@k = max trong k đáp án đầu."""
    if not rscores:
        return 0.0
    best_so_far: list[float] = []
    running = 0.0
    for value in rscores:
        running = max(running, value)
        best_so_far.append(running)
    total = 0.0
    for cutoff in BANDS:
        idx = min(cutoff, len(best_so_far)) - 1
        total += best_so_far[idx] if idx >= 0 else 0.0
    return total / len(BANDS)


def kis_rscores(answers: Sequence[Any], gt: Sequence[tuple[str, int]], tol: int) -> list[float]:
    by_video: dict[str, list[int]] = {}
    for video, frame in gt:
        by_video.setdefault(video, []).append(int(frame))
    out = []
    for a in answers:
        frames = by_video.get(a.video_id)
        hit = bool(frames) and any(abs(a.frame_idx - f) <= tol for f in frames)
        out.append(1.0 if hit else 0.0)
    return out


def trake_rscores(
    rows: Sequence[dict[str, Any]], alternatives: dict[str, list[list[Any]]], tol: int
) -> list[float]:
    """Điểm một dòng TRAKE: 0 nếu sai video, ngược lại là tỉ lệ event khớp.

    Một truy vấn có thể có nhiều phương án GT (nhóm ghi hai chuỗi hợp lệ), lấy max.
    """
    prepared = []
    for alt in alternatives.values():
        events = sorted(alt, key=lambda e: int(e[0]))
        prepared.append((events[0][1], [int(e[2]) for e in events]))
    out = []
    for row in rows:
        best = 0.0
        for video, frames in prepared:
            if row["video_id"] != video or not frames:
                continue
            got = row["frames"]
            hits = sum(
                1
                for j, gt_frame in enumerate(frames)
                if j < len(got) and abs(int(got[j]) - gt_frame) <= tol
            )
            best = max(best, hits / len(frames))
        out.append(best)
    return out


def first_hit_rank(rscores: Sequence[float]) -> int | None:
    for i, value in enumerate(rscores, start=1):
        if value > 0:
            return i
    return None


# ---------------------------------------------------------------------------
# Chạy bộ sinh trên một truy vấn đã cache
# ---------------------------------------------------------------------------


def score_query(record: dict[str, Any], params: AnswerGenParams, tols: Sequence[int]) -> dict[str, Any]:
    groups = record.get("groups") or []
    kind = record.get("kind")
    if kind == "trake":
        # N comes from the statement, exactly as the console sends it — inferring
        # it from the chains would let a query whose chains are all partial score
        # against rows the organiser would reject.
        rows = generate_trake_rows(
            record.get("trake_sequences") or [],
            params,
            event_count=record.get("n_events") or len(record.get("trake_events") or []) or None,
        )
        per_tol = {
            tol: trake_rscores(rows, record.get("alternatives") or {}, tol) for tol in tols
        }
        n_answers = len(rows)
        top = [
            {"video_id": r["video_id"], "frames": r["frames"][:6]} for r in rows[:3]
        ]
    else:
        answers = generate_answers(groups, params)
        gt = [(v, int(f)) for v, f in (record.get("gt") or [])]
        per_tol = {tol: kis_rscores(answers, gt, tol) for tol in tols}
        n_answers = len(answers)
        top = [
            {"video_id": a.video_id, "frame_idx": a.frame_idx, "kind": a.kind}
            for a in answers[:3]
        ]
    return {
        "query_id": record.get("query_id"),
        "kind": kind,
        "n_answers": n_answers,
        "final": {tol: final_from_rscores(r) for tol, r in per_tol.items()},
        "first_hit": {tol: first_hit_rank(r) for tol, r in per_tol.items()},
        "best_rscore": {tol: (max(r) if r else 0.0) for tol, r in per_tol.items()},
        "top": top,
    }


def score_query_with_head(
    record: dict[str, Any], params: AnswerGenParams, tols: Sequence[int], head: int
) -> dict[str, Any]:
    """Chấm khi `head` vị trí đầu đã bị người chấm tay chiếm.

    Mô phỏng đúng luồng thật: người thao tác chọn frame họ tin, rồi bộ sinh điền
    phần còn lại. Lấy chính top-`head` của bộ sinh làm phần chấm tay là mô phỏng
    sát nhất — đo trên bài nộp thật, frame người chọn thường trùng đúng thứ bộ
    sinh xếp đầu.

    Ý nghĩa của phép đo nằm ở thế giới "người chọn SAI": lúc đó mọi dòng cùng
    video với họ đều vô giá trị, và câu hỏi là danh sách còn lại có thoát ra
    video khác đủ nhanh không.
    """
    groups = record.get("groups") or []
    plain = generate_answers(groups, params)
    picked = plain[:head]
    rest = generate_answers(
        groups,
        params,
        limit=max(0, 100 - len(picked)),
        taken=[(a.video_id, a.frame_idx) for a in picked],
    )
    answers = [*picked, *rest]
    gt = [(v, int(f)) for v, f in (record.get("gt") or [])]
    per_tol = {tol: kis_rscores(answers, gt, tol) for tol in tols}
    return {
        "query_id": record.get("query_id"),
        "kind": record.get("kind"),
        "n_answers": len(answers),
        "final": {tol: final_from_rscores(r) for tol, r in per_tol.items()},
        "first_hit": {tol: first_hit_rank(r) for tol, r in per_tol.items()},
        "best_rscore": {tol: (max(r) if r else 0.0) for tol, r in per_tol.items()},
        "top": [{"video_id": a.video_id, "frame_idx": a.frame_idx, "kind": a.kind} for a in answers[:3]],
        "videos_at_5": len({a.video_id for a in answers[:5]}),
        "videos_at_20": len({a.video_id for a in answers[:20]}),
    }


# ---------------------------------------------------------------------------
# Đánh giá cả tập
# ---------------------------------------------------------------------------

_RECORDS: list[dict[str, Any]] = []
#: >0 means "score with this many positions already spent by a person".
_HEAD = 0


def _init_worker(records: list[dict[str, Any]], head: int = 0) -> None:
    global _RECORDS, _HEAD
    _RECORDS = records
    _HEAD = head


def evaluate(
    records: Sequence[dict[str, Any]], params: AnswerGenParams, tols: Sequence[int] = TOLERANCES
) -> dict[str, Any]:
    per_query = [score_query(r, params, tols) for r in records]
    return _aggregate(per_query, tols)


def _aggregate(per_query: Sequence[dict[str, Any]], tols: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(per_query), "per_query": list(per_query)}
    for tol in tols:
        finals = [q["final"][tol] for q in per_query]
        ranks = [q["first_hit"][tol] for q in per_query]
        out[f"tol{tol}"] = {
            "final": statistics.mean(finals) if finals else 0.0,
            "hit@1": sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0.0,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / len(ranks) if ranks else 0.0,
            "hit@20": sum(1 for r in ranks if r and r <= 20) / len(ranks) if ranks else 0.0,
            "hit@100": sum(1 for r in ranks if r and r <= 100) / len(ranks) if ranks else 0.0,
        }
    return out


def _sweep_task(args: tuple[dict[str, Any], tuple[int, ...]]) -> float:
    """Mục tiêu tối ưu: Final trung bình trên NHIỀU mức dung sai.

    Không ai biết trước bề rộng đoạn [s, e] của BTC. Tối ưu ở một mức duy nhất sẽ
    chọn ra bộ tham số chỉ đúng cho giả định đó — dung sai chặt thì thuật toán học
    cách liệt kê keyframe, dung sai rộng thì học cách rải epsilon. Lấy trung bình
    trên cả thang buộc bộ tham số phải chịu được cả hai.
    """
    params_dict, tols = args
    params = AnswerGenParams.from_dict(params_dict)
    scored = [
        score_query_with_head(r, params, tols, _HEAD) if _HEAD else score_query(r, params, tols)
        for r in _RECORDS
    ]
    if not scored:
        return 0.0
    return statistics.mean(
        statistics.mean(q["final"][tol] for q in scored) for tol in tols
    )


# ---------------------------------------------------------------------------
# Không gian dò tham số
# ---------------------------------------------------------------------------

GRID: dict[str, list[Any]] = {
    "pool_depth": [100, 200, 300, 400, 600, 1000, 1500],
    "max_videos": [20, 40, 60, 100],
    "dedup_radius_s": [0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0],
    "anchor_support_weight": [0.0, 0.1, 0.25, 0.5, 1.0],
    "offset_decay": [0.4, 0.5, 0.6, 0.72, 0.85, 0.95],
    "novelty_sigma_s": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    "novelty_floor": [0.0, 0.02, 0.1, 0.25],
    "novelty_mode": ["product", "min"],
    "ambiguous_anchor_bonus": [0, 1, 2, 4],
    "ambiguous_offset_penalty": [0, 1, 2],
    "min_answer_gap_frames": [0, 5, 10, 25, 50],
    "snap_offsets": [True, False],
    "snap_radius_frames": [0, 10, 20, 30, 60, 120],
    "temperatures": [
        (0.02, 0.02, 0.05, 0.10, 0.20),
        (0.05, 0.05, 0.10, 0.20, 0.40),
        (0.05, 0.10, 0.20, 0.40, 0.80),
        (0.10, 0.10, 0.20, 0.40, 0.80),
        (0.02, 0.05, 0.15, 0.35, 1.00),
        (0.01, 0.03, 0.08, 0.20, 0.60),
        (0.20, 0.20, 0.40, 0.80, 1.60),
    ],
    "band_anchors": [
        (1, 2, 4, 7, 10),
        (1, 3, 6, 10, 14),
        (1, 2, 3, 5, 8),
        (2, 3, 5, 8, 12),
        (1, 4, 8, 12, 20),
        (1, 2, 6, 14, 30),
        (1, 3, 8, 20, 40),
        (2, 5, 12, 25, 40),
    ],
    "band_offsets": [
        (0, 1, 2, 3, 5),
        (0, 0, 1, 3, 5),
        (0, 1, 3, 5, 7),
        (1, 2, 3, 4, 5),
        (0, 2, 4, 6, 7),
        (1, 1, 2, 4, 7),
        (0, 0, 0, 2, 5),
    ],
    # Thang epsilon lấy từ phân vị sai số truy xuất đo trên chính tập này
    # (mode calib): trung vị ~15 frame, Q75 ~30, đuôi vài trăm.
    "offsets": [
        (25, 60, 120, 240, 480),
        (12, 25, 50, 100, 200),
        (10, 30, 75, 150, 300, 600),
        (5, 12, 25, 50, 100, 250, 600),
        (5, 10, 20, 40, 80, 160, 320),
        (25, 75, 150, 300),
        (50, 125, 250, 500, 1000),
        (8, 20, 45, 90, 180, 360, 720),
        (3, 8, 15, 30, 60, 125, 250, 500),
    ],
}


def random_params(rng: random.Random, base: AnswerGenParams) -> AnswerGenParams:
    """Một điểm khởi đầu ngẫu nhiên trong lưới, để leo đồi không mắc kẹt một chỗ."""
    fields = {}
    for key, values in GRID.items():
        value = rng.choice(values)
        fields[key] = tuple(value) if isinstance(value, (list, tuple)) else value
    return replace(base, **fields).validated()


def coordinate_descent(
    records: Sequence[dict[str, Any]],
    start: AnswerGenParams,
    tols: Sequence[int],
    passes: int,
    workers: int,
    order: Sequence[str] | None = None,
) -> tuple[AnswerGenParams, float, list[dict[str, Any]]]:
    """Line search từng tham số, lặp lại vài vòng. Đơn giản nhưng đọc được: mỗi
    bước ghi lại tham số nào đổi, đổi thành gì và Final nhích lên bao nhiêu."""
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(list(records),)
    ) as pool:
        return _descend(pool, start, tols, passes, order, verbose=True)


#: Mọi cấu hình đã chấm trong phiên tìm kiếm, để phân tích biên sau đó.
SEARCH_LOG: list[tuple[dict[str, Any], float]] = []


def _descend(pool, start, tols, passes, order=None, verbose=True):
    """Leo đồi theo từng toạ độ trong một process pool đã mở sẵn."""
    best = start.validated()
    trail: list[dict[str, Any]] = []
    keys = list(order or GRID.keys())
    best_score = pool.submit(_sweep_task, (best.to_dict(), tuple(tols))).result()
    SEARCH_LOG.append((best.to_dict(), best_score))
    if verbose:
        print(f"khởi điểm objective={best_score:.4f}", flush=True)
    for p in range(1, passes + 1):
        improved = False
        for key in keys:
            candidates = [
                replace(
                    best, **{key: tuple(v) if isinstance(v, (list, tuple)) else v}
                ).validated()
                for v in GRID[key]
            ]
            scores = list(
                pool.map(_sweep_task, [(c.to_dict(), tuple(tols)) for c in candidates], chunksize=1)
            )
            SEARCH_LOG.extend((c.to_dict(), sc) for c, sc in zip(candidates, scores))
            top_i = max(range(len(scores)), key=lambda i: (scores[i], -i))
            if scores[top_i] > best_score + 1e-9:
                trail.append(
                    {
                        "pass": p,
                        "param": key,
                        "from": getattr(best, key),
                        "to": getattr(candidates[top_i], key),
                        "final": scores[top_i],
                        "delta": scores[top_i] - best_score,
                    }
                )
                if verbose:
                    print(
                        f"  [pass {p}] {key}: {getattr(best, key)} → {getattr(candidates[top_i], key)}"
                        f"  objective {best_score:.4f} → {scores[top_i]:.4f}",
                        flush=True,
                    )
                best = candidates[top_i]
                best_score = scores[top_i]
                improved = True
        if not improved:
            if verbose:
                print(f"  [pass {p}] không cải thiện thêm — dừng", flush=True)
            break
    return best, best_score, trail


def multistart_search(
    records: Sequence[dict[str, Any]],
    start: AnswerGenParams,
    tols: Sequence[int],
    passes: int,
    workers: int,
    restarts: int,
    seed: int = 20260820,
) -> tuple[AnswerGenParams, float, list[dict[str, Any]]]:
    """Leo đồi nhiều lần từ nhiều điểm khởi đầu và giữ kết quả tốt nhất.

    Leo đồi một lần từ tham số mặc định dừng ở cực trị địa phương — lần chạy đầu
    tiên đứng lại sau hai vòng. Khởi động lại ngẫu nhiên là cách rẻ nhất để biết
    cực trị đó có phải là điểm tốt nhất trong lưới hay không.
    """
    rng = random.Random(seed)
    starts = [start.validated()] + [random_params(rng, start) for _ in range(restarts)]
    best, best_score, best_trail = None, -1.0, []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(list(records),)
    ) as pool:
        for i, init in enumerate(starts):
            label = "mặc định" if i == 0 else f"ngẫu nhiên #{i}"
            print(f"\n--- khởi động {label} ---", flush=True)
            params, score, trail = _descend(pool, init, tols, passes, verbose=True)
            print(f"    → objective={score:.4f}", flush=True)
            if score > best_score:
                best, best_score, best_trail = params, score, trail
    assert best is not None
    return best, best_score, best_trail


def robust_params(
    log: Sequence[tuple[dict[str, Any], float]], base: AnswerGenParams, top_n: int = 400
) -> tuple[AnswerGenParams, dict[str, Any]]:
    """Chọn từng tham số theo biên (marginal) của các cấu hình tốt nhất.

    Với 21 truy vấn, chênh 0,01 điểm chỉ là một câu nhảy một mốc — bám lấy đúng
    cấu hình cao nhất là học thuộc tập phát triển. Ở đây mỗi tham số được chọn
    độc lập theo giá trị có điểm trung bình cao nhất trong nhóm cấu hình dẫn đầu,
    nên một giá trị chỉ thắng nếu nó tốt trên NHIỀU cấu hình khác nhau.
    """
    # Deduplicate first. A coordinate descent re-scores its current best on every
    # line search, so the raw log's leaders are the SAME configuration dozens of
    # times over — a marginal computed on that measures one point, not a trend.
    unique: dict[tuple, tuple[dict[str, Any], float]] = {}
    for cfg, score in log:
        key = tuple(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(cfg.items())
        )
        unique.setdefault(key, (cfg, score))
    ranked = sorted(unique.values(), key=lambda item: -item[1])[:top_n]
    fields: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for key, values in GRID.items():
        buckets: dict[Any, list[float]] = {}
        for cfg, score in ranked:
            raw = cfg.get(key)
            if raw is None:
                continue
            token = tuple(raw) if isinstance(raw, list) else raw
            buckets.setdefault(token, []).append(score)
        if not buckets:
            continue
        stats = {
            token: (statistics.mean(scores), len(scores)) for token, scores in buckets.items()
        }
        # Một giá trị chỉ được chọn nếu nó xuất hiện đủ nhiều trong nhóm dẫn đầu;
        # ngưỡng tỉ lệ theo kích thước nhóm chứ không phải một con số cố định.
        floor = max(3, len(ranked) // 40)
        eligible = {t: v for t, v in stats.items() if v[1] >= floor} or stats
        token = max(eligible, key=lambda t: (eligible[t][0], eligible[t][1]))
        fields[key] = token
        report[key] = {
            "chosen": list(token) if isinstance(token, tuple) else token,
            "mean": round(eligible[token][0], 4),
            "n": eligible[token][1],
            "alternatives": sorted(
                (
                    {
                        "value": list(t) if isinstance(t, tuple) else t,
                        "mean": round(v[0], 4),
                        "n": v[1],
                    }
                    for t, v in stats.items()
                ),
                key=lambda d: -d["mean"],
            )[:4],
        }
    return replace(base, **fields).validated(), report


# ---------------------------------------------------------------------------
# Kiểm chứng chéo
# ---------------------------------------------------------------------------


def cross_validate(
    records: Sequence[dict[str, Any]],
    tols: Sequence[int],
    passes: int,
    workers: int,
    restarts: int,
    folds: int = 3,
    repeats: int = 2,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Dò tham số trên k-1 phần, chấm trên phần còn lại.

    Con số duy nhất trả lời được câu "bộ tham số này có tổng quát hoá không, hay
    chỉ vừa khít 21 câu đã có". So sánh với chính DEFAULT_PARAMS trên cùng các
    phần held-out để biết phần cải thiện nào là thật.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = list(range(len(records)))
        rng.shuffle(order)
        for fold in range(folds):
            held = {order[i] for i in range(fold, len(order), folds)}
            train = [r for i, r in enumerate(records) if i not in held]
            test = [r for i, r in enumerate(records) if i in held]
            if not train or not test:
                continue
            tuned, train_score, _ = multistart_search(
                train, DEFAULT_PARAMS, tols, passes, workers, restarts, seed + repeat * 10 + fold
            )
            _init_worker(test)
            tuned_test = _sweep_task((tuned.to_dict(), tuple(tols)))
            base_test = _sweep_task((DEFAULT_PARAMS.to_dict(), tuple(tols)))
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_objective": train_score,
                    "test_tuned": tuned_test,
                    "test_default": base_test,
                    "params": tuned.to_dict(),
                }
            )
            print(
                f"  repeat {repeat} fold {fold}: train={train_score:.4f} "
                f"test_tuned={tuned_test:.4f} test_default={base_test:.4f}",
                flush=True,
            )
    return {
        "folds": rows,
        "mean_test_tuned": statistics.mean(r["test_tuned"] for r in rows) if rows else 0.0,
        "mean_test_default": statistics.mean(r["test_default"] for r in rows) if rows else 0.0,
        "mean_train": statistics.mean(r["train_objective"] for r in rows) if rows else 0.0,
    }


# ---------------------------------------------------------------------------
# Hiệu chỉnh thang epsilon
# ---------------------------------------------------------------------------


def calibrate(records: Sequence[dict[str, Any]], params: AnswerGenParams) -> dict[str, Any]:
    """|frame truy xuất − GT| trên tập phát triển, theo hai cách đọc.

    `anchor_error`  : mỏ neo mạnh nhất của video GT lệch bao xa — chính là sai số
                      mà các điểm dịch ±eps phải phủ.
    `nearest_error` : frame gần nhất trong toàn bộ frame đã truy xuất của video GT
                      — chặn dưới, cho biết thang eps có thể nhỏ đến đâu.
    """
    anchor_errors: list[int] = []
    nearest_errors: list[int] = []
    detail = []
    for record in records:
        if record.get("kind") == "trake":
            continue
        gt = [(v, int(f)) for v, f in (record.get("gt") or [])]
        gt_videos = {v for v, _ in gt}
        pools = build_pools(record.get("groups") or [], params)
        pool = next((p for p in pools if p.video_id in gt_videos), None)
        if pool is None:
            detail.append({"query_id": record.get("query_id"), "reason": "GT video không có trong pool"})
            continue
        frames = [f for v, f in gt if v == pool.video_id]
        best_anchor = min(
            (abs(pool.anchors[0].frame_idx - f) for f in frames), default=None
        )
        any_anchor = min(
            (abs(a.frame_idx - f) for a in pool.anchors for f in frames), default=None
        )
        group = next(g for g in record["groups"] if g["video_id"] == pool.video_id)
        nearest = min(
            (
                abs(int(fr["frame_idx"]) - f)
                for fr in group["frames"]
                if fr.get("frame_idx") is not None
                for f in frames
            ),
            default=None,
        )
        if best_anchor is not None:
            anchor_errors.append(best_anchor)
        if nearest is not None:
            nearest_errors.append(nearest)
        detail.append(
            {
                "query_id": record.get("query_id"),
                "gt_video": pool.video_id,
                "video_rank_in_pool": pools.index(pool) + 1,
                "top_anchor_error": best_anchor,
                "best_anchor_error": any_anchor,
                "nearest_frame_error": nearest,
                "n_anchors": len(pool.anchors),
                "ambiguous": pool.ambiguous,
            }
        )
    pct = (25, 50, 75, 90, 97)
    return {
        "n_with_gt_video": len(anchor_errors),
        "percentiles": list(pct),
        "top_anchor_error": retrieval_error_percentiles(anchor_errors, pct),
        "nearest_frame_error": retrieval_error_percentiles(nearest_errors, pct),
        "detail": detail,
    }


# ---------------------------------------------------------------------------


def load_cache(path: Path, kinds: set[str]) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [
        r
        for r in data["queries"]
        if not r.get("error") and r.get("kind") in kinds and (r.get("groups") or r.get("trake_sequences"))
    ]
    return records


def print_table(result: dict[str, Any], tols: Sequence[int]) -> None:
    print(f"n = {result['n']} truy vấn")
    print(f"{'tol(frame)':>11} {'Final':>8} {'hit@1':>7} {'hit@5':>7} {'hit@20':>7} {'hit@100':>8}")
    for tol in tols:
        row = result[f"tol{tol}"]
        print(
            f"{tol:>11} {row['final']:>8.4f} {row['hit@1']:>7.2%} {row['hit@5']:>7.2%}"
            f" {row['hit@20']:>7.2%} {row['hit@100']:>8.2%}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DIR / "pools_infoshotpp_L_auto.json"))
    ap.add_argument("--mode", default="eval", choices=["eval", "sweep", "calib", "cv", "taken"])
    ap.add_argument("--kinds", default="kis")
    ap.add_argument(
        "--tol",
        default="0,12,25,50,100",
        help="các mức dung sai (frame) gộp lại thành mục tiêu tối ưu",
    )
    ap.add_argument("--report-tol", type=int, default=25, help="mức dung sai in chi tiết")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--restarts", type=int, default=0, help="số điểm khởi đầu ngẫu nhiên")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--head", type=int, default=0, help="số vị trí đầu do người chấm tay chiếm")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--params", default="", help="file JSON tham số khởi điểm")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    tols = tuple(int(t) for t in str(args.tol).split(",") if t.strip() != "")
    records = load_cache(Path(args.cache), kinds)
    print(f"{len(records)} truy vấn từ {args.cache} (kinds={sorted(kinds)})", flush=True)

    params = DEFAULT_PARAMS
    if args.params:
        params = AnswerGenParams.from_dict(json.loads(Path(args.params).read_text(encoding="utf-8")))

    if args.mode == "calib":
        result = calibrate(records, params)
        print(json.dumps({k: v for k, v in result.items() if k != "detail"}, ensure_ascii=False, indent=2))
        for d in result["detail"]:
            print(" ", json.dumps(d, ensure_ascii=False))
        return

    if args.mode == "taken":
        head = max(1, args.head)
        print(f"Mô phỏng {head} vị trí đầu do người chấm tay chiếm.\n")
        print(f"{'phạt video đã chiếm':>20} {'objective':>10}  " +
              "  ".join(f"F@{t}" for t in TOLERANCES) + "   v@5")
        for penalty in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            trial = replace(params, taken_video_penalty=penalty).validated()
            scored = [score_query_with_head(r, trial, TOLERANCES, head) for r in records]
            obj = statistics.mean(
                statistics.mean(q["final"][t] for q in scored) for t in tols
            )
            cells = "  ".join(
                f"{statistics.mean(q['final'][t] for q in scored):.3f}" for t in TOLERANCES
            )
            v5 = statistics.mean(q["videos_at_5"] for q in scored)
            print(f"{penalty:>20} {obj:>10.4f}  {cells}   {v5:.2f}")
        return

    if args.mode == "cv":
        result = cross_validate(
            records, tols, args.passes, args.workers, max(1, args.restarts), seed=args.seed
        )
        print(
            f"\ntest (tuned)   = {result['mean_test_tuned']:.4f}\n"
            f"test (mặc định) = {result['mean_test_default']:.4f}\n"
            f"train           = {result['mean_train']:.4f}"
        )
        dest = Path(args.out) if args.out else RUNS_DIR / "answer_gen_cv.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {dest}")
        return

    if args.mode == "eval":
        result = evaluate(records, params, sorted(set(TOLERANCES) | set(tols)))
        print_table(result, TOLERANCES)
        print()
        rt = args.report_tol
        print(f"objective (trung bình Final trên tol={list(tols)}) = "
              f"{statistics.mean(result[f'tol{t}']['final'] for t in tols if f'tol{t}' in result):.4f}")
        for q in sorted(result["per_query"], key=lambda q: -q["final"][rt]):
            print(
                f"  {q['query_id']:<22} final@{rt}={q['final'][rt]:.2f}"
                f" first_hit={q['first_hit'][rt]}  top1={q['top'][0] if q['top'] else None}"
            )
        if args.out:
            Path(args.out).write_text(
                json.dumps({"params": params.to_dict(), "result": result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return

    if args.restarts:
        best, score, trail = multistart_search(
            records, params, tols, args.passes, args.workers, args.restarts, args.seed
        )
    else:
        best, score, trail = coordinate_descent(records, params, tols, args.passes, args.workers)
    print(f"\nobjective (Final trung bình trên tol={list(tols)}) = {score:.4f}")
    robust, marginal = robust_params(SEARCH_LOG, DEFAULT_PARAMS)
    _init_worker(list(records))
    robust_score = _sweep_task((robust.to_dict(), tuple(tols)))
    print(f"cấu hình theo biên (marginal của {len(SEARCH_LOG)} lần chấm) = {robust_score:.4f}")
    if robust_score >= score - 1e-9:
        print("→ dùng cấu hình theo biên (không kém hơn, và ít bám tập phát triển hơn)")
        best, score = robust, robust_score
    else:
        print("→ giữ cấu hình tốt nhất tuyệt đối")
    print(json.dumps(best.to_dict(), ensure_ascii=False, indent=2))
    result = evaluate(records, best, TOLERANCES)
    print()
    print_table(result, TOLERANCES)
    tag = "-".join(str(t) for t in tols)
    dest = Path(args.out) if args.out else RUNS_DIR / f"answer_gen_sweep_{'_'.join(sorted(kinds))}_tol{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "cache": args.cache,
                "kinds": sorted(kinds),
                "tolerances": list(tols),
                "best_final": score,
                "params": best.to_dict(),
                "robust_params": robust.to_dict(),
                "robust_final": robust_score,
                "marginal": marginal,
                "n_configs_scored": len(SEARCH_LOG),
                "trail": [
                    {**t, "from": list(t["from"]) if isinstance(t["from"], tuple) else t["from"],
                     "to": list(t["to"]) if isinstance(t["to"], tuple) else t["to"]}
                    for t in trail
                ],
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"→ {dest}")


if __name__ == "__main__":
    main()
