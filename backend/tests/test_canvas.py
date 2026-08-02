"""Canvas query logic: text generation, one-to-one assignment, scoring modes."""
from __future__ import annotations

import pytest

from app.canvas import (
    CanvasObject,
    canvas_to_queries,
    grid7_of,
    match_canvas,
    nearest_palette_color,
    parse_canvas,
    relation_between,
    zone_of,
)


def detection(label, bbox, conf=0.9, color=None, reliable=True):
    x1, y1, x2, y2 = bbox
    return {
        "label": label,
        "canonical_label": label,
        "conf": conf,
        "bbox_norm": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "center_norm": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
        "position": "center",
        "dominant_color": color,
        "color_names": [color] if color else [],
        "color_reliable": reliable if color else False,
    }


CAR_LEFT = {"id": "q1", "label": "car", "bbox": [0.05, 0.52, 0.35, 0.86], "color": "red"}
PERSON_RIGHT = {"id": "q2", "label": "person", "bbox": [0.66, 0.35, 0.82, 0.86], "color": "blue"}


class TestParseAndDescribe:
    def test_degenerate_boxes_are_dropped(self):
        canvas = parse_canvas({"objects": [CAR_LEFT, {"id": "q9", "label": "person", "bbox": [0.5, 0.5, 0.5, 0.5]}]})

        assert [obj.id for obj in canvas.objects] == ["q1"]

    def test_bbox_is_normalized_and_ordered(self):
        canvas = parse_canvas({"objects": [{"label": "car", "bbox": [0.9, 0.8, 0.2, 0.1]}]})

        assert canvas.objects[0].bbox == (0.2, 0.1, 0.9, 0.8)

    def test_color_is_dropped_for_a_class_without_color_data(self):
        # OD only extracts colour for COLOR_CLASSES; keeping "red crowd" would
        # filter every frame out instead of just ignoring an unusable hint.
        canvas = parse_canvas({"objects": [{"label": "crowd", "bbox": [0.1, 0.1, 0.4, 0.5], "color": "red"}]})

        assert canvas.objects[0].color is None

    def test_unknown_color_is_dropped(self):
        canvas = parse_canvas({"objects": [{"label": "car", "bbox": [0.1, 0.1, 0.4, 0.5], "color": "turquoise"}]})

        assert canvas.objects[0].color is None

    def test_nearest_palette_color_snaps_free_rgb(self):
        assert nearest_palette_color((250, 10, 10)) == "red"
        assert nearest_palette_color((15, 25, 85)) == "navy"

    def test_zone_and_grid_match_the_od_quantisation(self):
        assert zone_of(0.1, 0.1) == "top_left"
        assert zone_of(0.5, 0.5) == "center"
        assert zone_of(0.9, 0.9) == "bottom_right"
        assert grid7_of(0.0, 0.0) == "r0c0"
        assert grid7_of(0.99, 0.99) == "r6c6"

    def test_queries_mention_color_label_and_position(self):
        canvas = parse_canvas({"objects": [CAR_LEFT, PERSON_RIGHT]})

        queries = canvas_to_queries(canvas)

        assert queries, "canvas with objects must produce PE text"
        joined = " ".join(queries).lower()
        assert "red car" in joined
        assert "blue person" in joined
        assert "lower left" in joined

    def test_action_text_leads_the_query_list(self):
        canvas = parse_canvas({"objects": [CAR_LEFT], "action_text": "a man walks past the car"})

        assert canvas_to_queries(canvas)[0].startswith("a man walks past the car")

    def test_relation_between_uses_the_dominant_axis(self):
        left = CanvasObject("a", "car", (0.05, 0.5, 0.35, 0.9))
        right = CanvasObject("b", "person", (0.66, 0.5, 0.82, 0.9))
        top = CanvasObject("c", "sign", (0.4, 0.02, 0.6, 0.2))

        assert relation_between(left, right) == "left_of"
        assert relation_between(right, left) == "right_of"
        assert relation_between(top, left) == "above"


class TestMatching:
    def test_layout_that_matches_scores_far_above_a_mirrored_one(self):
        canvas = parse_canvas({"objects": [CAR_LEFT, PERSON_RIGHT]})
        correct = [
            detection("car", (0.06, 0.50, 0.36, 0.88), color="red"),
            detection("person", (0.65, 0.34, 0.84, 0.88), color="blue"),
        ]
        mirrored = [
            detection("car", (0.66, 0.50, 0.94, 0.88), color="red"),
            detection("person", (0.10, 0.34, 0.28, 0.88), color="blue"),
        ]

        good = match_canvas(canvas, correct)
        bad = match_canvas(canvas, mirrored)

        assert good["coverage"] == 1.0
        assert good["score"] > bad["score"]

    def test_two_drawn_objects_never_share_one_detection(self):
        canvas = parse_canvas({
            "objects": [
                {"id": "p1", "label": "person", "bbox": [0.10, 0.30, 0.30, 0.90]},
                {"id": "p2", "label": "person", "bbox": [0.70, 0.30, 0.90, 0.90]},
            ]
        })
        single_person = [detection("person", (0.12, 0.30, 0.30, 0.90))]

        result = match_canvas(canvas, single_person)

        assert len(result["matches"]) == 1
        assert result["missing"] == ["p2"]
        matched_ids = [match["object_id"] for match in result["matches"]]
        assert matched_ids == ["p1"]

    def test_two_people_drawn_and_two_detected_are_matched_pairwise(self):
        canvas = parse_canvas({
            "objects": [
                {"id": "p1", "label": "person", "bbox": [0.10, 0.30, 0.30, 0.90]},
                {"id": "p2", "label": "person", "bbox": [0.70, 0.30, 0.90, 0.90]},
            ]
        })
        two = [
            detection("person", (0.68, 0.30, 0.88, 0.90)),
            detection("person", (0.12, 0.30, 0.32, 0.90)),
        ]

        result = match_canvas(canvas, two)

        assert result["coverage"] == 1.0
        pairs = {match["object_id"]: match["bbox_norm"]["x1"] for match in result["matches"]}
        assert pairs["p1"] < pairs["p2"]

    def test_wrong_label_is_never_matched_by_position_alone(self):
        canvas = parse_canvas({"objects": [CAR_LEFT]})

        result = match_canvas(canvas, [detection("bus", (0.05, 0.52, 0.35, 0.86), color="red")])

        assert result["matches"] == []
        assert result["score"] == 0.0

    def test_missing_required_object_is_penalised_not_deleted(self):
        canvas = parse_canvas({"objects": [CAR_LEFT, {**PERSON_RIGHT, "required": True}]})
        optional = parse_canvas({"objects": [CAR_LEFT, {**PERSON_RIGHT, "required": False}]})
        car_only = [detection("car", (0.06, 0.50, 0.36, 0.88), color="red")]

        required_result = match_canvas(canvas, car_only)
        optional_result = match_canvas(optional, car_only)

        assert 0 < required_result["score"] < optional_result["score"]

    def test_color_mismatch_lowers_but_keeps_the_match(self):
        canvas = parse_canvas({"objects": [CAR_LEFT]})
        right_color = match_canvas(canvas, [detection("car", (0.06, 0.5, 0.36, 0.88), color="red")])
        wrong_color = match_canvas(canvas, [detection("car", (0.06, 0.5, 0.36, 0.88), color="white")])

        assert wrong_color["coverage"] == 1.0
        assert wrong_color["score"] < right_color["score"]
        assert wrong_color["matches"][0]["color_ok"] is False

    def test_unreliable_color_does_not_decide_the_match(self):
        canvas = parse_canvas({"objects": [CAR_LEFT]})
        unreliable = match_canvas(
            canvas, [detection("car", (0.06, 0.5, 0.36, 0.88), color="white", reliable=False)]
        )
        wrong_but_reliable = match_canvas(
            canvas, [detection("car", (0.06, 0.5, 0.36, 0.88), color="white", reliable=True)]
        )

        assert unreliable["score"] > wrong_but_reliable["score"]

    def test_rough_mode_tolerates_a_shifted_box_better_than_precise(self):
        payload = {"objects": [CAR_LEFT]}
        shifted = [detection("car", (0.20, 0.42, 0.52, 0.80), color="red")]

        rough = match_canvas(parse_canvas({**payload, "mode": "rough"}), shifted)
        precise = match_canvas(parse_canvas({**payload, "mode": "precise"}), shifted)

        assert rough["score"] > precise["score"]

    def test_excluded_label_present_is_a_penalty_not_a_filter(self):
        canvas = parse_canvas({"objects": [CAR_LEFT], "exclude_labels": ["person"]})
        frames = [
            detection("car", (0.06, 0.50, 0.36, 0.88), color="red"),
            detection("person", (0.70, 0.30, 0.90, 0.90), conf=0.8),
        ]

        result = match_canvas(canvas, frames)

        assert result["excluded_hits"] == ["person"]
        assert result["score"] > 0

    def test_low_confidence_detections_are_ignored(self):
        canvas = parse_canvas({"objects": [CAR_LEFT]})

        result = match_canvas(canvas, [detection("car", (0.06, 0.5, 0.36, 0.88), conf=0.05, color="red")])

        assert result["matches"] == []

    def test_empty_canvas_scores_zero(self):
        assert match_canvas(parse_canvas({}), [detection("car", (0, 0, 1, 1))])["score"] == 0.0


@pytest.mark.asyncio
async def test_canvas_search_ranks_the_intended_frame_first(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)
    result = await service.search({"canvas": {"objects": [CAR_LEFT, PERSON_RIGHT]}})

    assert result["groups"], "canvas search must return candidates in mock mode"
    top_frames = [frame["submit_keyframe_id"] for frame in result["groups"][0]["frames"]]
    assert "K01/K01_V001/002" in top_frames
    assert result["canvas"]["queries_en"], "PE text must be generated from the canvas"


@pytest.mark.asyncio
async def test_canvas_search_attaches_overlay_evidence(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)
    result = await service.search({"canvas": {"objects": [CAR_LEFT, PERSON_RIGHT]}})

    frames = [frame for group in result["groups"] for frame in group["frames"]]
    target = next(frame for frame in frames if frame["submit_keyframe_id"] == "K01/K01_V001/002")
    layout = next(item for item in target["evidence"] if item["type"] == "object_layout")

    assert layout["coverage"] == 1.0
    assert {match["label"] for match in layout["matches"]} == {"car", "person"}
    assert all("bbox_norm" in match for match in layout["matches"])


@pytest.mark.asyncio
async def test_empty_canvas_returns_a_warning_not_an_error(settings):
    from app.services.canvas_service import CanvasService

    result = await CanvasService(settings).search({"canvas": {"objects": []}})

    assert result["groups"] == []
    assert result["warnings"]


PNG_1PX = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestRasterChannel:
    def test_only_inline_image_bytes_are_accepted(self):
        # The string is forwarded to the Kaggle encoder, so a URL here would turn
        # that server into a fetcher for whatever the browser asked for.
        assert parse_canvas({"image": PNG_1PX}).image == PNG_1PX
        assert parse_canvas({"image": "https://evil.example/x.png"}).image is None
        assert parse_canvas({"image": "data:text/html;base64,PHNjcmlwdD4="}).image is None
        assert parse_canvas({}).image is None


@pytest.mark.asyncio
async def test_canvas_raster_adds_its_own_channel(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)
    result = await service.search({"canvas": {"objects": [CAR_LEFT], "image": PNG_1PX}})

    assert result["canvas"]["has_image"] is True
    channels = {channel for group in result["groups"] for frame in group["frames"] for channel in frame["channels"]}
    assert "canvas_image" in channels


@pytest.mark.asyncio
async def test_canvas_without_raster_never_calls_the_image_encoder(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("encode_image called without a canvas raster")

    service.search_service.pe.encode_image = fail
    result = await service.search({"canvas": {"objects": [CAR_LEFT]}})

    assert result["canvas"]["has_image"] is False
    channels = {channel for group in result["groups"] for frame in group["frames"] for channel in frame["channels"]}
    assert "canvas_image" not in channels


@pytest.mark.asyncio
async def test_missing_encode_image_route_is_reported_as_a_warning(settings):
    from app.adapters.pe_encoder import PeImageEncoderMissing
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)

    async def missing(*args, **kwargs):
        raise PeImageEncoderMissing("PE server chưa có /encode-image")

    service.search_service.pe.encode_image = missing
    result = await service.search({"canvas": {"objects": [CAR_LEFT], "image": PNG_1PX}})

    # The other channels still answer; only the raster is degraded.
    assert result["groups"], "a dead raster channel must not empty the result"
    assert any("/encode-image" in warning for warning in result["warnings"])
