from app.scoring import audio_score_multiplier, speech_score_multiplier


def test_speech_low_confidence_strongly_demoted():
    assert speech_score_multiplier("low", "body") < speech_score_multiplier("high", "body")
    assert speech_score_multiplier("mid", "body") < speech_score_multiplier("high", "body")


def test_speech_intro_preview_demoted():
    assert speech_score_multiplier("high", "intro") < speech_score_multiplier("high", "body")
    assert speech_score_multiplier("high", "preview") < speech_score_multiplier("high", "body")


def test_audio_top1_stoplist_drops():
    assert audio_score_multiplier(
        audio_stoplist_hit=True, top1_is_stoplisted=True, caption_quality="useful"
    ) == 0.0


def test_audio_generic_caption_demoted():
    useful = audio_score_multiplier(
        audio_stoplist_hit=False, top1_is_stoplisted=False, caption_quality="useful"
    )
    generic = audio_score_multiplier(
        audio_stoplist_hit=False, top1_is_stoplisted=False, caption_quality="generic"
    )
    vasr = audio_score_multiplier(
        audio_stoplist_hit=False, top1_is_stoplisted=False, caption_quality="vietnamese_asr"
    )
    assert generic < useful
    assert vasr < generic


def test_audio_top1_match_boosted():
    base = audio_score_multiplier(
        audio_stoplist_hit=False, top1_is_stoplisted=False, caption_quality="useful"
    )
    boosted = audio_score_multiplier(
        audio_stoplist_hit=False,
        top1_is_stoplisted=False,
        caption_quality="useful",
        matched_on_top1=True,
    )
    assert boosted > base
