"""Evidence-aware stopping; confidence alone never certifies temporal tasks."""


def _supported(frame, constraints, threshold):
    if frame.get("probability") is None or frame["probability"] < threshold:
        return False
    for constraint in constraints:
        score = frame.get("constraint_scores", {}).get(constraint.id)
        # Calibration is fitted on candidate relevance, not constraint labels.
        # Apply the stopping threshold to the calibrated candidate probability;
        # retain a separate raw majority-evidence guard for every constraint.
        if not score or score["supported"] <= .5 or score["supported"] <= max(score["unknown"], score["refuted"]):
            return False
    return True


def can_stop(board, threshold: float, margin: float, *, require_constraints: bool = True) -> bool:
    ranked = board.ranked()
    if not ranked or board.query_type == "AVS":
        return False  # One supported item cannot complete ad-hoc retrieval.
    top = ranked[0]
    checks = board.constraints if require_constraints else [c for c in board.constraints if c.id in {"query", "answer"}]
    applicable = [c for c in checks if c.event is None or c.event == top.get("event")]
    if not _supported(top, applicable, threshold):
        return False
    if board.query_type == "QA" and not top.get("answer"):
        return False
    if board.query_type == "TRAKE":
        events = [c for c in board.constraints if c.event is not None]
        if len(events) < 2:
            return False
        previous = -1.0
        for constraint in events:
            # Certify the exact chain the ranked output presents, never a
            # lower-ranked alternative that the evaluator/operator won't use.
            row = next((f for f in ranked if f["video_id"] == top["video_id"]
                        and f.get("event") == constraint.event), None)
            event_checks = [c for c in checks if c.event is None or c.event == constraint.event]
            if row is None or not _supported(row, event_checks, threshold):
                return False
            t = row.get("time") if row.get("time") is not None else row.get("pts_time")
            if t is None or t <= previous:
                return False
            previous = t
        # Other events from this same sequence are not competitors.
        competitors = [f for f in ranked[1:] if f["video_id"] != top["video_id"]]
    else:
        competitors = [f for f in ranked[1:] if f["video_id"] != top["video_id"] or
                       f.get("pts_time") is None or top.get("pts_time") is None or
                       abs(f["pts_time"] - top["pts_time"]) > 8]
    other = max((f["probability"] for f in competitors if f.get("probability") is not None), default=0.0)
    return top["probability"] - other >= margin
