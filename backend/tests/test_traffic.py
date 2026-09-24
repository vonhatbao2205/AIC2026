"""Traffic-camera / race-stage cues: parsing, resolution and the pushed-down filter."""
import json

import pytest

from app.adapters.elastic_client import ElasticClient
from app.scope import FrameFilter, elastic_filter_clause, milvus_filter_expr
from app.services.search_service import SearchService, _apply_filters
from app.traffic import (
    Camera,
    CameraVideo,
    TrafficCatalog,
    load_catalog,
    parse_dates,
    parse_stage,
    parse_time,
    resolve_traffic,
)
from app.types import ChannelHit, FusedFrame


def hit(video_id, n, category=None):
    category = category or video_id.split("-")[0].split("_")[0]
    return ChannelHit(
        channel="image_pe", submit_keyframe_id=f"{category}/{video_id}/{n:03d}", video_id=video_id,
        keyframe_n=n, score=1.0, rank=0,
    )


def small_catalog():
    """Two junctions sharing a street; one video lost its banner date."""
    cameras = [
        Camera("a", "Nguyễn Trãi – Cống Quỳnh", "Nguyen Trai - Cong Quynh", ("Nguyễn Trãi", "Cống Quỳnh"), ("N092-V001", "N092-V002")),
        Camera("b", "Nguyễn Thị Minh Khai – Cống Quỳnh", "NTMK - CongQuynh", ("Nguyễn Thị Minh Khai", "Cống Quỳnh"), ("N063-V001",)),
    ]
    videos = {
        # 19:00-19:02 over 30 keyframes
        "N092-V001": CameraVideo("N092-V001", "a", 30, ("2026-06-15",), ("19:00:00", "19:02:59"), ((1140, 1, 10), (1141, 11, 20), (1142, 21, 30))),
        # 11:00-11:01 over 20 keyframes
        "N092-V002": CameraVideo("N092-V002", "a", 20, ("2026-06-10",), ("11:00:00", "11:01:59"), ((660, 1, 10), (661, 11, 20))),
        "N063-V001": CameraVideo("N063-V001", "b", 10, (), ("19:01:00", "19:01:59"), ((1141, 1, 10),)),
    }
    return TrafficCatalog(cameras, videos)


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "window"),
        [
            ("lúc 19 giờ", (19 * 60 - 5, 19 * 60 + 64)),
            ("khoảng 7 giờ tối", (19 * 60 - 15, 19 * 60 + 74)),
            ("19h05", (1143, 1147)),
            ("lúc 19:05:30", (1143, 1147)),
            ("7:30 pm", (1168, 1172)),
            ("từ 19:02 đến 19:05", (1140, 1147)),
            ("vào buổi sáng", (5 * 60, 11 * 60 + 59)),
            ("luc 7 gio toi", (19 * 60 - 5, 19 * 60 + 64)),
        ],
    )
    def test_time_windows(self, text, window):
        cue = parse_time(text)
        assert (cue.start, cue.end) == window

    def test_accents_decide_between_look_alike_words(self):
        # "tôi" (I) is not "tối" (evening); "gió" (wind) is not "giờ" (hour).
        assert parse_time("7 giờ tôi đi làm").start == 7 * 60 - 5
        assert parse_time("gió cấp 7") is None
        # "chặng sau" is "the next leg", not stage 6 ("chặng sáu").
        assert parse_stage("chặng sau đó") is None
        assert parse_stage("chặng sáu") == 6
        # "3 thắng 1" is a score, "15 tháng 6" a date.
        assert parse_dates("3 thắng 1") == []
        assert parse_dates("15 tháng 6") == [(None, 6, 15)]

    @pytest.mark.parametrize(
        ("text", "dates"),
        [
            ("ngày 15/6/2026", [(2026, 6, 15)]),
            ("10.Jun 2026", [(2026, 6, 10)]),
            ("June 15, 2026", [(2026, 6, 15)]),
            ("2026-06-15", [(2026, 6, 15)]),
            ("15 máy bay", []),
        ],
    )
    def test_dates(self, text, dates):
        assert parse_dates(text) == dates

    def test_stage(self):
        assert parse_stage("tay đua ở chặng 6") == 6
        assert parse_stage("chặng thứ 3") == 3
        assert parse_stage("chặng cuối") == 12
        assert parse_stage("stage 11") == 11
        assert parse_stage("chàng trai") is None


class TestResolution:
    def test_junction_date_and_time_narrow_to_keyframes(self):
        result = resolve_traffic("ngã tư Nguyễn Trãi Cống Quỳnh lúc 19:01 ngày 15/6", catalog=small_catalog())
        assert [camera.id for camera in result.cameras] == ["a"]
        assert result.dates == ["2026-06-15"]
        # 19:01 +/- 2 minutes covers minutes 1139..1143 -> the whole of N092-V001.
        assert result.frames.window_map == {"N092-V001": ()}
        assert result.videos == 1 and result.keyframes == 30

    def test_one_street_keeps_every_camera_on_it(self):
        result = resolve_traffic("xe buýt trên đường Cống Quỳnh", catalog=small_catalog())
        assert {camera.id for camera in result.cameras} == {"a", "b"}
        assert set(result.frames.window_map) == {"N092-V001", "N092-V002", "N063-V001"}

    def test_time_becomes_keyframe_intervals(self):
        catalog = small_catalog()
        # 11:00 +/- 2 -> minutes 658..662: all of N092-V002 (660, 661).
        assert resolve_traffic("Nguyễn Trãi - Cống Quỳnh 11:00", catalog=catalog).frames.window_map == {"N092-V002": ()}
        # 19:04 +/- 2 -> minutes 1142..1146: only the last minute of N092-V001.
        result = resolve_traffic("Nguyễn Trãi - Cống Quỳnh lúc 19:04", catalog=catalog)
        assert result.frames.window_map == {"N092-V001": ((21, 30),)}
        assert result.keyframes == 10
        assert result.frames.allows("N092-V001", 25) and not result.frames.allows("N092-V001", 20)

    def test_time_that_no_camera_recorded_is_dropped(self):
        catalog = small_catalog()
        # One street, 19:00 +/- 2: both 19h videos, not the 11h one.
        result = resolve_traffic("Cống Quỳnh vào 19:00", catalog=catalog)
        assert result.frames.window_map == {"N092-V001": (), "N063-V001": ()}
        # 19:10 +/- 2 misses every recording of the junction: time dropped, camera kept.
        result = resolve_traffic("Nguyễn Trãi Cống Quỳnh lúc 19:10", catalog=catalog)
        assert result.time is None and "19:08–19:12" in result.warnings[0]
        assert set(result.frames.window_map) == {"N092-V001", "N092-V002"}

    def test_date_no_camera_recorded_is_ignored_not_emptying(self):
        result = resolve_traffic("đường 3/2 kẹt xe", catalog=small_catalog())
        assert result.frames is None
        assert "03/02" in result.warnings[0]

    def test_unknown_banner_date_cannot_rule_a_video_out(self):
        result = resolve_traffic("Cống Quỳnh ngày 10/6", catalog=small_catalog())
        assert set(result.frames.window_map) == {"N092-V002", "N063-V001"}

    def test_stage_constrains_only_the_race(self):
        result = resolve_traffic("tay đua áo vàng ở chặng 6", catalog=small_catalog())
        assert result.frames.families == ("S01-",)
        assert result.frames.allows("S01-V006", 5000)
        assert not result.frames.allows("S01-V007", 1)
        assert result.frames.allows("L23_V001", 1)  # the other race is untouched
        assert result.frames.allows("N092-V001", 1)

    def test_off_and_empty(self):
        assert resolve_traffic("Nguyễn Trãi Cống Quỳnh", mode="off", catalog=small_catalog()).frames is None
        assert resolve_traffic("một người đàn ông", catalog=small_catalog()).frames is None

    def test_real_catalog_covers_every_camera_video(self):
        catalog = load_catalog()
        assert len(catalog.videos) == 298 and len(catalog.cameras) == 98
        for video in catalog.videos.values():
            runs = video.minutes
            assert runs[0][1] == 1 and runs[-1][2] == video.keyframes
            assert all(a[2] + 1 == b[1] for a, b in zip(runs, runs[1:]))

    @pytest.mark.parametrize(
        ("query", "folders"),
        [
            ("ngã tư Nguyễn Trãi - Cống Quỳnh", {"N092", "N093"}),
            ("camera NTMK – Đinh Tiên Hoàng", {"N088", "N089"}),
            ("Cách Mạng Tháng Tám giao Nguyễn Đình Chiểu", {"N004"}),
            ("vòng xoay Ngã 6 Cộng Hòa", {"N075", "N076"}),
            ("NKKN - Lý Tự Trọng", {"N069"}),
        ],
    )
    def test_real_catalog_names_junctions(self, query, folders):
        result = resolve_traffic(query)
        assert {video_id.split("-")[0] for video_id in result.frames.window_map} == folders


class TestFrameFilter:
    frames = FrameFilter(families=("N",), windows=(("N001-V001", ()), ("N002-V003", ((5, 9), (20, 22)))))

    def test_allows(self):
        assert self.frames.allows("L21_V001", 1) and self.frames.allows("M01_V001", 1)
        assert self.frames.allows("N001-V001", 999)
        assert self.frames.allows("N002-V003", 21) and not self.frames.allows("N002-V003", 10)
        assert not self.frames.allows("N003-V001", 1)

    def test_milvus_expression(self):
        assert self.frames.milvus_expr() == (
            '((not (video_id like "N%")) or video_id in ["N001-V001"] or '
            '(video_id == "N002-V003" and ((keyframe_n >= 5 and keyframe_n <= 9) or '
            "(keyframe_n >= 20 and keyframe_n <= 22))))"
        )
        combined = milvus_filter_expr(("N001", "L21"), self.frames)
        assert combined.startswith('(video_id like "N001-%" or video_id like "L21_%") and ((not')

    def test_elastic_clause(self):
        clause = elastic_filter_clause((), self.frames)
        should = clause["bool"]["should"]
        assert should[0] == {"bool": {"must_not": [{"prefix": {"video_id": "N"}}]}}
        assert should[1] == {"terms": {"video_id": ["N001-V001"]}}
        assert "N002-V003" in json.dumps(should[2])
        both = elastic_filter_clause(("N001",), self.frames)
        assert len(both["bool"]["filter"]) == 2

    def test_rejects_unsafe_values(self):
        with pytest.raises(ValueError):
            FrameFilter(families=("N",), windows=(('N001-V001" or 1==1', ()),))
        with pytest.raises(ValueError):
            FrameFilter(families=("N",), windows=(("L21_V001", ()),))
        with pytest.raises(ValueError):
            FrameFilter(families=("N",), windows=(("N001-V001", ((0, 3),)),))

    def test_post_filter(self):
        hits = [hit("N001-V001", 3), hit("N002-V003", 10), hit("N002-V003", 21), hit("L21_V001", 7)]
        kept = _apply_filters(hits, {}, (), self.frames)
        assert [(h.video_id, h.keyframe_n) for h in kept] == [("N001-V001", 3), ("N002-V003", 21), ("L21_V001", 7)]


class TestElasticOcr:
    @pytest.mark.asyncio
    async def test_overlay_text_is_found_below_scene_text(self, settings, monkeypatch):
        client = ElasticClient(settings)
        client.mock = False
        bodies = []

        async def fake_search(index, body):
            bodies.append(body)
            return {"hits": {"hits": []}}

        monkeypatch.setattr(client, "_search", fake_search)
        await client.search_ocr(["tin mới"], [], exact_phrases=["HIGHLANDS"])
        clauses = bodies[0]["query"]["bool"]["must"][0]["dis_max"]["queries"]
        text = json.dumps(clauses)
        # Every text_nfc clause is limited to the L documents (no `profile`).
        nfc = [c for c in clauses if "text_nfc" in json.dumps(c)]
        assert nfc and all(c["bool"]["must_not"] == [{"exists": {"field": "profile"}}] for c in nfc)
        # Ticker / banner / HUD are searched, with boosts below every scene clause.
        overlay = [c for c in clauses if any(f in json.dumps(c) for f in ("text_ticker", "text_banner", "text_hud"))]
        assert overlay and "text_ticker_fold" in text
        boosts = [next(iter(next(iter(c.values())).values()))["boost"] for c in overlay]
        assert max(boosts) < 5.0

    @pytest.mark.asyncio
    async def test_clock_without_seconds_matches_the_whole_minute(self, settings, monkeypatch):
        client = ElasticClient(settings)
        client.mock = False
        bodies = []

        async def fake_search(index, body):
            bodies.append(body)
            return {"hits": {"hits": []}}

        monkeypatch.setattr(client, "_search", fake_search)
        await client.search_ocr([], [], hour=8, clock="8:05")
        should = bodies[0]["query"]["bool"]["filter"][0]["bool"]["should"]
        assert {"prefix": {"clock": "08:05:"}} in should and {"prefix": {"clock": "8:05:"}} in should
        assert {"prefix": {"race_time": "8:05:"}} in should

    @pytest.mark.asyncio
    async def test_frames_ride_in_the_filter(self, settings, monkeypatch):
        client = ElasticClient(settings)
        client.mock = False
        bodies = []

        async def fake_search(index, body):
            bodies.append(body)
            return {"hits": {"hits": []}}

        monkeypatch.setattr(client, "_search", fake_search)
        frames = FrameFilter(families=("N",), windows=(("N001-V001", ()),))
        await client.search_speech(["xe buýt"], frames=frames)
        assert bodies[0]["query"]["bool"]["filter"] == [frames.elastic_clause()]


class TestService:
    def test_profile_gate(self, settings):
        btc = SearchService(settings.for_retrieval_database("btc"))
        assert btc.resolve_traffic("auto", query="Nguyễn Trãi Cống Quỳnh").frames is None
        infoshot = SearchService(settings.for_retrieval_database("infoshotpp"))
        result = infoshot.resolve_traffic("auto", query="Nguyễn Trãi Cống Quỳnh", hints=["lúc 19 giờ"])
        assert result.frames is not None and result.time is not None
        assert infoshot.resolve_traffic("off", query="Nguyễn Trãi Cống Quỳnh").frames is None

    @pytest.mark.asyncio
    async def test_overlay_enrichment(self, settings):
        service = SearchService(settings.for_retrieval_database("infoshotpp"))

        async def overlays(ids):
            return {"N092/N092-V001/005": {"clock": "19:00:41", "banner_date": "2026-06-15"}}

        service.elastic.get_ocr_overlays = overlays
        frames = [
            FusedFrame("N092/N092-V001/005", "N092-V001", 5, None, 1.0),
            FusedFrame("L21/L21_V001/005", "L21_V001", 5, None, 0.5),
        ]
        await service._enrich_overlay(frames)
        assert frames[0].overlay["clock"] == "19:00:41"
        assert frames[0].overlay["camera"] == load_catalog().cameras[load_catalog().videos["N092-V001"].camera].label
        assert frames[1].overlay == {}

    @pytest.mark.asyncio
    async def test_search_reports_the_filter(self, settings):
        service = SearchService(settings.for_retrieval_database("infoshotpp"))
        result = await service.search({"query": "ngã tư Nguyễn Trãi Cống Quỳnh, chặng 6", "image_models": ["pe"]})
        assert result["traffic"]["active"] is True
        assert result["traffic"]["race_stage"] == 6
        assert {camera["id"] for camera in result["traffic"]["cameras"]} == {"nguyentraicongquynh1", "nguyentraicongquynh2"}
        off = await service.search({"query": "ngã tư Nguyễn Trãi Cống Quỳnh", "traffic": "off", "image_models": ["pe"]})
        assert off["traffic"]["active"] is False
