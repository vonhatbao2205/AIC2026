"""Per-channel score adjustments (demotions) shared by real and mock adapters.

These encode the quality policy from the handoff docs so they are unit-testable
in isolation and applied identically regardless of data source.
"""
from __future__ import annotations

# Speech confidence_bucket multipliers.
SPEECH_BUCKET_MULT = {"high": 1.0, "mid": 0.75, "low": 0.25, "missing": 0.5}
# segment_role multipliers (intro/preview demoted as temporal anchors).
SPEECH_ROLE_MULT = {"body": 1.0, "intro": 0.7, "preview": 0.6}

# Audio caption_quality multipliers.
AUDIO_CAPTION_MULT = {"useful": 1.0, "none": 0.9, "generic": 0.5, "vietnamese_asr": 0.4}


def speech_score_multiplier(confidence_bucket: str | None, segment_role: str | None) -> float:
    bucket = SPEECH_BUCKET_MULT.get((confidence_bucket or "missing"), 0.5)
    role = SPEECH_ROLE_MULT.get((segment_role or "body"), 1.0)
    return bucket * role


def audio_score_multiplier(
    *,
    audio_stoplist_hit: bool,
    top1_is_stoplisted: bool,
    caption_quality: str | None,
    matched_on_top1: bool = False,
) -> float:
    """Demote/drop stoplisted audio; trust top1 more than secondary tags.

    Returns 0.0 to indicate the hit should be dropped.
    """
    if top1_is_stoplisted:
        return 0.0
    mult = 1.0
    if audio_stoplist_hit:
        mult *= 0.3
    mult *= AUDIO_CAPTION_MULT.get((caption_quality or "none"), 0.9)
    if matched_on_top1:
        mult *= 1.2  # reward a top1_label match over secondary-tag matches
    return mult
