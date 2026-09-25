"""Search scope: the folder catalogue, the topic heuristic, and the push-down."""
from types import SimpleNamespace

import pytest

from app import scope
from app.services.search_service import SearchService, _apply_filters


# ---- catalogue --------------------------------------------------------
def test_profiles_expose_the_folders_they_actually_hold():
    btc = scope.profile_categories("btc")
    infoshot = scope.profile_categories("infoshotpp")

    assert btc[:10] == tuple(f"L{n}" for n in range(21, 31))
    assert btc[10:] == tuple(f"K{n:02d}" for n in range(1, 21))
    # InfoShot++ was re-extracted from the L-side and also holds data batch 2;
    # offering K01-K20 there (or batch 2 on BTC) would be a filter that can only
    # ever return nothing.
    assert infoshot[:10] == tuple(f"L{n}" for n in range(21, 31))
    assert infoshot[10:20] == tuple(f"M{n:02d}" for n in range(1, 11))
    assert infoshot[20:120] == tuple(f"N{n:03d}" for n in range(1, 101))
    assert infoshot[120:] == ("S01",)
    assert not [cat for cat in infoshot if cat.startswith("K")]
    assert not [cat for cat in btc if cat[0] in "MNS"]


def test_catalogue_labels_every_folder_with_its_programme():
    body = scope.catalogue("btc")
    labels = {item["category"]: item["label_vi"] for item in body["categories"]}

    assert "Nấu ăn" in labels["L26"]
    assert "Đua xe đạp" in labels["L23"]
    assert "HTV7" in labels["K01"] and "HTV9" in labels["K02"]
    assert {group["id"] for group in body["groups"]} == {"L", "K"}
    english = {item["category"]: item["label_en"] for item in body["categories"]}
    assert english["L26"] == "Cooking — HTV Online"
    assert set(english) == set(labels)
    assert all(group["label_en"] for group in body["groups"])


def test_catalogue_hides_topics_with_no_folder_in_this_profile():
    infoshot = {topic["topic_id"]: topic for topic in scope.catalogue("infoshotpp")["topics"]}
    btc = {topic["topic_id"]: topic for topic in scope.catalogue("btc")["topics"]}
    # What must NOT survive is a category the profile lacks inside a topic: K-only
    # news folders on InfoShot++, batch-2 folders on BTC.
    assert infoshot["news"]["categories"] == ["L21", "L22", *(f"M{n:02d}" for n in range(1, 11))]
    assert infoshot["cycling"]["categories"] == ["L23", "S01"]
    assert btc["cycling"]["categories"] == ["L23"]
    # The traffic-camera topic has no folder at all on BTC, so it is hidden there.
    assert "traffic" in infoshot and "traffic" not in btc


# ---- topic heuristic --------------------------------------------------
@pytest.mark.parametrize(
    ("query", "topic_id", "folder"),
    [
        ("Cảnh người đầu bếp rưới nước sốt vào nồi", "cooking", "L26"),
        ("Tìm đoạn đua xe đạp, một tay đua áo xanh vượt lên", "cycling", "L23"),
        ("Lân quay vòng trên cột số 4 rồi tiếp đất", "lion_dance", "L24"),
        ("Đây là câu 3 trong bài tập vận dụng, đáp án của bài là C", "exam", "L25"),
        ("Đoạn clip thu hoạch dứa ở miền Tây", "mekong", "L28"),
        ("Buổi trao quà từ thiện của một câu lạc bộ", "positive_energy", "L30"),
        ("Bản tin thời sự 60 giây của HTV9", "news", "K02"),
    ],
)
def test_programme_words_route_to_their_folder(query, topic_id, folder):
    resolved = scope.resolve_scope({"mode": "auto"}, query=query, retrieval_database="btc")

    assert topic_id in {match.topic_id for match in resolved.matches}
    assert folder in resolved.categories
    assert folder in resolved.strict_categories
    display = resolved.to_dict()
    assert display["reason_en"]
    assert all(match["label_en"] for match in display["matched_topics"])


def test_a_topic_cue_never_excludes_the_open_subject_folders():
    """The 60-second bulletin reports on cooking too, and L30 accepts anything.

    Measured on the ground-truth queries: a cooking clip really does live in K14
    and L30, and a cycling report in K19, so a speciality cue narrows the L-side
    without ever ruling those out.
    """
    resolved = scope.resolve_scope(
        {"mode": "auto"}, query="một đoạn video nấu ăn món tôm", retrieval_database="btc"
    )

    assert "L26" in resolved.categories
    assert {"K14", "K19", "L21", "L22", "L30"} <= set(resolved.categories)
    # The other speciality programmes are what the cue actually rules out.
    assert not {"L23", "L24", "L25", "L27"} & set(resolved.categories)
    # ...and the strict set the console offers as one click is the topic alone.
    assert resolved.strict_categories == ("L26",)


def test_matching_is_accent_sensitive_when_the_query_has_diacritics():
    """Folding both sides equates distinct words and picked the wrong folder.

    "vẫy tay đưa" folds onto the cycling term "tay đua" and "cụ lão" onto the
    Mekong term "cù lao"; both fired on unrelated K-side and L30 queries.
    """
    assert scope.match_topics("Người cảnh sát vẫy tay đưa nghi phạm vào phòng") == ()
    assert scope.match_topics("quyển sách có ảnh chân dung một cụ lão in trắng đen") == ()


def test_an_unaccented_query_still_matches_through_folding():
    """An operator typing "nau an" has no diacritics to be sensitive about."""
    resolved = scope.resolve_scope(
        {"mode": "auto"}, query="canh nguoi dau bep dang nau an", retrieval_database="btc"
    )

    assert resolved.strict_categories == ("L26",)


def test_no_topic_cue_leaves_the_search_unfiltered():
    resolved = scope.resolve_scope(
        {"mode": "auto"}, query="một người mặc áo trắng ngồi giữa hai người áo đen",
        retrieval_database="btc",
    )

    assert resolved.categories == ()
    assert resolved.active is False


# ---- modes ------------------------------------------------------------
def test_manual_mode_takes_exactly_what_was_ticked():
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": ["l26", " K01 ", "L23"]},
        query="bất kỳ",
        retrieval_database="btc",
    )

    assert resolved.categories == ("L23", "L26", "K01")


def test_manual_mode_drops_folders_the_profile_does_not_hold():
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": ["L26", "K01", "K02"]},
        query="",
        retrieval_database="infoshotpp",
    )

    assert resolved.categories == ("L26",)


def test_selecting_everything_is_normalized_to_no_filter():
    """A filter that cannot exclude anything must not reach the adapters, where
    it would cost a `like` clause per folder on every channel."""
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": list(scope.profile_categories("infoshotpp"))},
        query="",
        retrieval_database="infoshotpp",
    )

    assert resolved.categories == ()
    assert resolved.active is False


def test_all_mode_and_an_unknown_mode_both_search_everything():
    for spec in ({"mode": "all"}, {"mode": "nonsense"}, {}, None):
        assert scope.resolve_scope(spec, query="nấu ăn", retrieval_database="btc").categories == ()


def test_infoshotpp_topics_reach_the_batch2_folders():
    def auto(query):
        return scope.resolve_scope({"mode": "auto"}, query=query, retrieval_database="infoshotpp")

    cycling = auto("Tìm đoạn đua xe đạp, một tay đua áo xanh vượt lên")
    assert {"L23", "S01"} <= set(cycling.strict_categories)
    # The M bulletins are open-subject (M06 replays the race); cameras are not.
    assert "M06" in cycling.categories and "N001" not in cycling.categories

    traffic = auto("camera giao thông ở ngã tư, một xe buýt rẽ trái")
    assert set(traffic.strict_categories) == set(scope.N_CATEGORIES)
    assert "M01" in traffic.categories and "S01" not in traffic.categories

    cooking = auto("Cảnh người đầu bếp rưới nước sốt vào nồi")
    assert "L26" in cooking.categories
    assert not {"S01", "N001"} & set(cooking.categories)


# ---- push-down filters ------------------------------------------------
def test_milvus_expression_matches_on_the_video_id_prefix():
    assert scope.milvus_filter_expr(("L26",)) == '(video_id like "L26_%")'
    assert scope.milvus_filter_expr(("L26", "K01")) == (
        '(video_id like "L26_%" or video_id like "K01_%")'
    )
    assert scope.milvus_filter_expr(()) == ""


def test_hyphenated_batch2_folders_filter_on_their_own_separator():
    """N001-V001 / S01-V001 would never match a "N001_" prefix."""
    assert scope.milvus_filter_expr(("M01", "N001", "S01")) == (
        '(video_id like "M01_%" or video_id like "N001-%" or video_id like "S01-%")'
    )
    assert scope.elastic_filter_clause(("N100",))["bool"]["should"] == [{"prefix": {"video_id": "N100-"}}]


def test_a_category_that_is_not_catalogue_shaped_never_reaches_a_query_string():
    """The expression is string-built, so anything with a quote or a wildcard in
    it is dropped rather than escaped."""
    assert scope.milvus_filter_expr(('L26" or video_id like "K',)) == ""
    assert scope.elastic_filter_clause(("L2*",)) is None


def test_elastic_clause_is_a_should_over_prefixes():
    clause = scope.elastic_filter_clause(("L26", "L27"))

    assert clause == {
        "bool": {
            "should": [
                {"prefix": {"video_id": "L26_"}},
                {"prefix": {"video_id": "L27_"}},
            ],
            "minimum_should_match": 1,
        }
    }
    assert scope.elastic_filter_clause(()) is None


# ---- end to end through the service ----------------------------------
@pytest.mark.asyncio
async def test_search_only_returns_frames_from_the_selected_folders(settings):
    """Mock mode holds K01 and L26 videos; ticking L26 must drop the K01 ones."""
    svc = SearchService(settings)

    res = await svc.search(
        {"query": "bản tin", "scope": {"mode": "manual", "categories": ["L26"]}}
    )

    assert res["groups"], "expected the L26 video to survive the filter"
    assert {group["video_id"] for group in res["groups"]} == {"L26_V001"}
    assert res["scope"]["categories"] == ["L26"]
    assert res["scope"]["active"] is True


@pytest.mark.asyncio
async def test_search_reports_the_scope_it_applied(settings):
    svc = SearchService(settings)

    res = await svc.search({"query": "cảnh người đầu bếp nấu ăn", "scope": {"mode": "auto"}})

    reported = res["scope"]
    assert reported["mode"] == "auto"
    assert "L26" in reported["categories"]
    assert reported["strict_categories"] == ["L26"]
    assert [topic["topic_id"] for topic in reported["matched_topics"]] == ["cooking"]


@pytest.mark.asyncio
async def test_an_unscoped_search_is_unchanged(settings):
    svc = SearchService(settings)

    scoped = await svc.search({"query": "bản tin thời sự", "scope": {"mode": "all"}})
    plain = await svc.search({"query": "bản tin thời sự"})

    assert scoped["scope"]["active"] is False
    assert [g["video_id"] for g in scoped["groups"]] == [g["video_id"] for g in plain["groups"]]


def test_the_camera_and_race_groups_start_collapsed_in_the_picker():
    groups = {group["id"]: group for group in scope.catalogue("infoshotpp")["groups"]}
    assert [gid for gid, group in groups.items() if group["collapsed"]] == ["N", "S"]
    assert len(groups["N"]["categories"]) == 100


def test_the_race_folder_lists_its_twelve_stages():
    categories = {item["category"]: item for item in scope.catalogue("infoshotpp")["categories"]}

    stages = categories["S01"]["videos"]
    assert [stage["video_id"] for stage in stages] == [f"S01-V{n:03d}" for n in range(1, 13)]
    assert stages[5]["label_en"] == "Stage 6"
    assert "videos" not in categories["L23"]


# ---- single race stages ------------------------------------------------
def test_manual_mode_can_narrow_the_race_to_single_stages():
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": ["s01-v011", "L23", "S01-V006"]},
        query="",
        retrieval_database="infoshotpp",
    )

    assert resolved.categories == ("L23", "S01-V006", "S01-V011")
    assert scope.milvus_filter_expr(("S01-V006",)) == '(video_id like "S01-V006%")'
    assert scope.elastic_filter_clause(("S01-V006",))["bool"]["should"] == [{"prefix": {"video_id": "S01-V006"}}]


def test_a_stage_next_to_its_whole_folder_adds_nothing():
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": ["S01", "S01-V006"]}, query="", retrieval_database="infoshotpp"
    )

    assert resolved.categories == ("S01",)


def test_a_stage_the_profile_does_not_hold_is_dropped():
    resolved = scope.resolve_scope(
        {"mode": "manual", "categories": ["S01-V006", "N001-V001"]}, query="", retrieval_database="btc"
    )

    assert resolved.active is False


def test_frames_of_a_picked_stage_survive_the_post_filter():
    """The live post-filter compares the frame's folder; a stage must pass it too."""
    hit = lambda video_id: SimpleNamespace(  # noqa: E731
        video_id=video_id, keyframe_n=1, submit_keyframe_id=f"{video_id[:3]}/{video_id}/001"
    )
    hits = [hit("S01-V006"), hit("S01-V007"), hit("L23_V001")]

    kept = _apply_filters(hits, {}, ("S01-V006", "L23"))

    assert [h.video_id for h in kept] == ["S01-V006", "L23_V001"]
    assert scope.in_scope("S01-V006", {"S01-V006"}) and scope.in_scope("S01-V006", {"S01"})
    assert not scope.in_scope("S01-V007", {"S01-V006"})


def test_camera_folders_are_labelled_with_the_junction_they_film():
    labels = {item["category"]: item["label_en"] for item in scope.catalogue("infoshotpp")["categories"]}

    assert labels["N001"] == "An Dương Vương – Lê Hồng Phong"
    # N016 mixes several cameras; every junction in it is named.
    assert "Kỳ Đồng – Bà Huyện Thanh Quan" in labels["N016"]
    assert all(not labels[f"N{n:03d}"].startswith("Traffic camera") for n in range(1, 101))
    assert labels["S01"] == "Cycling — 2026 Television Cup (batch 2)"
