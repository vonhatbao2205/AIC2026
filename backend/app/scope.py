"""Search scope: which dataset folders (categories) a query is allowed to hit.

Two things live here.

1. The **catalogue**: which categories each retrieval profile actually holds, and
   what programme each one is. BTC carries the first-round corpus (L21-L30 +
   K01-K20); InfoShot++ was re-extracted from the L-side (L21-L30) and also holds
   data batch 2 (M01-M10 news, N001-N100 traffic cameras, S01 cycling).

2. The **topic heuristic**: the folders are not an arbitrary split, they are one
   programme each ("Nấu ăn của HTV Online" IS L26, "Đua xe đạp của HTV Thể thao"
   IS L23). A query naming what the video is about therefore already says which
   folders can hold the answer — the same read the query parser does for channels.

The heuristic narrows on format words only ("đầu bếp", "chặng đua"), never on
subject words ("tôm", "bánh"), and it never excludes the open-subject folders —
see `OPEN_SUBJECT_CATEGORIES`. Measured on the 104 ground-truth queries in
`TKIS/QA/TRAKE_queries.xlsx` it fires on 22 and puts the answer's folder in scope
on all 22; dropping the open-subject rule loses the answer on 6 of them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .identity import video_id_prefix
from .text_normalization import fold_vietnamese

#: L-side: the ten InfoShot++ programmes. K-side: the twenty BTC news batches.
L_CATEGORIES: tuple[str, ...] = tuple(f"L{n}" for n in range(21, 31))
K_CATEGORIES: tuple[str, ...] = tuple(f"K{n:02d}" for n in range(1, 21))
#: Data batch 2 (InfoShot++ only): HTV7 "60 giây" bulletins, one folder per fixed
#: traffic camera, and the 2026 Television Cup cycling race.
M_CATEGORIES: tuple[str, ...] = tuple(f"M{n:02d}" for n in range(1, 11))
N_CATEGORIES: tuple[str, ...] = tuple(f"N{n:03d}" for n in range(1, 101))
S_CATEGORIES: tuple[str, ...] = ("S01",)
BATCH2_CATEGORIES: tuple[str, ...] = (*M_CATEGORIES, *N_CATEGORIES, *S_CATEGORIES)
ALL_CATEGORIES: tuple[str, ...] = (*L_CATEGORIES, *K_CATEGORIES, *BATCH2_CATEGORIES)

#: The K batches sourced from HTV7; the rest of the K-side is HTV9.
_HTV7_BATCHES = frozenset({"K01", "K04", "K06", "K08", "K10", "K12", "K14", "K16", "K18", "K20"})

#: What each folder is, from the organisers' dataset notes. Shown in the UI so
#: the operator picks a programme rather than an opaque code.
CATEGORY_LABELS: dict[str, str] = {
    "L21": "Thời sự 60 giây — HTV9",
    "L22": "Thời sự 60 giây — HTV7",
    "L23": "Đua xe đạp — HTV Thể thao",
    "L24": "Múa lân — HTV Thể thao",
    "L25": "Ôn thi THPTQG — Thanh Niên",
    "L26": "Nấu ăn — HTV Online",
    "L27": "Khám phá Văn hóa Việt Nam — HTV Online",
    "L28": "Lưu vực sông Mekong — HTV Online",
    "L29": "Lưu vực sông Mekong (chương trình khác) — HTV Online",
    "L30": "Lan tỏa năng lượng tích cực — Tuổi Trẻ TV",
    **{cat: f"Thời sự 60 giây — {'HTV7' if cat in _HTV7_BATCHES else 'HTV9'}" for cat in K_CATEGORIES},
    **{cat: "Thời sự 60 giây — HTV7 (batch 2)" for cat in M_CATEGORIES},
    **{cat: f"Camera giao thông {cat} (batch 2)" for cat in N_CATEGORIES},
    "S01": "Đua xe đạp — Cúp Truyền hình 2026 (batch 2)",
}

CATEGORY_LABELS_EN: dict[str, str] = {
    "L21": "60-second news — HTV9",
    "L22": "60-second news — HTV7",
    "L23": "Cycling — HTV Sports",
    "L24": "Lion dance — HTV Sports",
    "L25": "National high school exam revision — Thanh Nien",
    "L26": "Cooking — HTV Online",
    "L27": "Exploring Vietnamese culture — HTV Online",
    "L28": "Mekong River basin — HTV Online",
    "L29": "Mekong River basin (alternate program) — HTV Online",
    "L30": "Spreading positive energy — Tuoi Tre TV",
    **{cat: f"60-second news — {'HTV7' if cat in _HTV7_BATCHES else 'HTV9'}" for cat in K_CATEGORIES},
    **{cat: "60-second news — HTV7 (batch 2)" for cat in M_CATEGORIES},
    **{cat: f"Traffic camera {cat} (batch 2)" for cat in N_CATEGORIES},
    "S01": "Cycling — 2026 Television Cup (batch 2)",
}
TOPIC_LABELS_EN = {
    "cooking": "Cooking", "cycling": "Cycling", "lion_dance": "Lion and dragon dance",
    "exam": "National high school exam revision", "culture": "Exploring Vietnamese culture",
    "mekong": "Mekong River basin / delta", "positive_energy": "Spreading positive energy",
    "news": "60-second news", "traffic": "Traffic cameras",
}
GROUP_LABELS_EN = {
    "L": "L21–L30 · themed programs (InfoShot++)",
    "K": "K01–K20 · 60-second news (BTC)",
    "M": "M01–M10 · 60-second news HTV7 (batch 2)",
    "N": "N001–N100 · traffic cameras (batch 2)",
    "S": "S01 · cycling (batch 2)",
}

#: Categories each retrieval profile can return. InfoShot++ has no K-side data,
#: and batch 2 was only ever extracted and embedded for InfoShot++, so offering a
#: folder a profile lacks would be a filter that always returns nothing.
PROFILE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "btc": (*L_CATEGORIES, *K_CATEGORIES),
    "infoshotpp": (*L_CATEGORIES, *BATCH2_CATEGORIES),
}

#: Folders whose programme has no fixed subject: the 60-second bulletin reports
#: on cooking, cycling and lion dance alike, and L30 is a user-submitted shorts
#: contest that accepts anything. A topic cue is evidence about which SPECIALITY
#: folder could hold the answer; it is never evidence against these. Measured on
#: the ground-truth queries, excluding them is exactly what loses the answer: a
#: cycling report really does live in K19, a cooking clip in K14 and L30. The
#: batch-2 M folders are the same HTV7 bulletin (M06 replays the S01 race).
OPEN_SUBJECT_CATEGORIES: tuple[str, ...] = ("L21", "L22", "L30", *K_CATEGORIES, *M_CATEGORIES)

#: Coarse grouping for the UI's "select a whole side" shortcut.
CATEGORY_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("L", "L21–L30 · chuyên đề (InfoShot++)", L_CATEGORIES),
    ("K", "K01–K20 · Thời sự 60 giây (BTC)", K_CATEGORIES),
    ("M", "M01–M10 · Thời sự 60 giây HTV7 (batch 2)", M_CATEGORIES),
    ("N", "N001–N100 · Camera giao thông (batch 2)", N_CATEGORIES),
    ("S", "S01 · Đua xe đạp (batch 2)", S_CATEGORIES),
)
#: Groups the picker shows as a single checkbox. The hundred N folders are one
#: street camera each and differ only by their code, so listing them one by one
#: buries every other folder under a hundred rows that say nothing.
COLLAPSED_GROUPS: frozenset[str] = frozenset({"N"})


@dataclass(frozen=True)
class Topic:
    """One programme subject and the words that name it.

    Keywords describe the programme FORMAT, not what it happens to film. "đầu
    bếp" and "công thức" belong to a cooking show; "tôm" and "bánh" do not —
    those appear just as often in a news report about a food festival, and
    including them made the heuristic pick L26 for six different K-side queries.
    """

    id: str
    label_vi: str
    categories: tuple[str, ...]
    keywords: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    Topic(
        id="cooking",
        label_vi="Nấu ăn",
        categories=("L26",),
        keywords=(
            "nấu ăn", "đầu bếp", "công thức", "nguyên liệu", "gia vị", "nước sốt",
            "nước chấm", "ướp", "vỉ nướng", "chảo dầu", "dạy nấu", "món ăn",
            "sơ chế", "khẩu phần", "cooking", "chef", "recipe", "ingredient",
            "seasoning", "marinate", "frying pan",
        ),
    ),
    Topic(
        id="cycling",
        label_vi="Đua xe đạp",
        categories=("L23", *S_CATEGORIES),
        keywords=(
            "đua xe đạp", "xe đạp", "tay đua", "cua rơ", "cuarơ", "chặng đua",
            "đoàn đua", "áo vàng", "peloton", "cycling", "cyclist", "bicycle",
            "bike race", "time trial",
        ),
    ),
    Topic(
        id="lion_dance",
        label_vi="Múa lân / lân sư rồng",
        categories=("L24",),
        keywords=(
            # Bare "lân" earns its place: across all 104 ground-truth queries it
            # occurs three times and all three are L24. Bare "rồng" does not —
            # it also names a dragon-dance news report (K08) and a dragon fish
            # (K18) — so only the compound "múa rồng" is listed.
            "lân", "múa lân", "múa rồng", "lân sư rồng", "mai hoa thung",
            "đầu lân", "ông địa", "lion dance", "dragon dance",
        ),
    ),
    Topic(
        id="exam",
        label_vi="Ôn thi THPTQG",
        categories=("L25",),
        keywords=(
            "ôn thi", "luyện thi", "thpt", "đề thi", "bài tập vận dụng",
            "trắc nghiệm", "đáp án của bài", "bài giảng", "exam revision",
            "multiple choice", "exam question",
        ),
    ),
    # L27 (văn hóa Việt Nam) and L28/L29 (lưu vực Mekong) are the same HTV Online
    # documentary family and their subjects overlap — a craft village in the delta
    # can be filed under either — so both topics keep all three folders in scope.
    Topic(
        id="culture",
        label_vi="Khám phá văn hóa Việt Nam",
        categories=("L27", "L28", "L29"),
        keywords=(
            "khám phá văn hóa", "làng nghề", "nghệ nhân", "đình thần",
            "lễ hội truyền thống", "di tích lịch sử", "craft village", "artisan",
            "traditional festival", "heritage site",
        ),
    ),
    Topic(
        id="mekong",
        label_vi="Lưu vực sông Mekong / miền Tây",
        categories=("L28", "L29", "L27"),
        keywords=(
            "mekong", "mê kông", "cửu long", "miền tây", "chợ nổi", "cù lao",
            "kênh rạch", "miệt vườn", "sông nước", "mekong delta",
            "floating market", "mekong river",
        ),
    ),
    Topic(
        id="positive_energy",
        label_vi="Lan tỏa năng lượng tích cực",
        categories=("L30",),
        keywords=(
            "năng lượng tích cực", "lan tỏa", "tuổi trẻ tv", "thiện nguyện",
            "từ thiện", "trao quà", "mạnh thường quân", "truyền cảm hứng",
            "positive energy", "charity", "volunteer",
        ),
    ),
    Topic(
        id="news",
        label_vi="Thời sự 60 giây",
        # L21/L22 are the InfoShot++ copies of the same bulletin the whole K-side
        # carries, and M01-M10 are more HTV7 editions of it, so a news cue must
        # keep all three.
        categories=("L21", "L22", *K_CATEGORIES, *M_CATEGORIES),
        keywords=(
            "thời sự", "bản tin", "60 giây", "htv7", "htv9", "đưa tin",
            "phóng sự", "tin tức", "phóng viên", "mẩu tin", "đoạn tin",
            "newscast", "news bulletin", "news report",
        ),
    ),
    # The N folders are fixed street cameras: no cuts, no presenter, a banner with
    # the junction and the clock. Only words naming that FORMAT are listed —
    # "xe máy" or "đèn đỏ" turn up in news reports just as often.
    Topic(
        id="traffic",
        label_vi="Camera giao thông",
        categories=N_CATEGORIES,
        keywords=(
            "camera giao thông", "camera an ninh", "camera quan sát", "camera cố định",
            "camera đường phố", "nút giao", "ngã tư", "giao lộ", "traffic camera",
            "cctv", "surveillance camera", "intersection",
        ),
    ),
)


@dataclass(frozen=True)
class TopicMatch:
    topic_id: str
    label_vi: str
    keywords: tuple[str, ...]
    categories: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedScope:
    """What the search will actually be restricted to.

    `categories` is empty when nothing is filtered (mode "all", or an auto pass
    that found no topic) — retrieval then runs against the whole profile.
    `strict_categories` is the narrower set the matched topics alone imply, which
    the console offers as a one-click "only these folders" chip for an operator
    who is sure the answer is not in a news bulletin.
    """

    mode: str
    categories: tuple[str, ...]
    strict_categories: tuple[str, ...] = ()
    matches: tuple[TopicMatch, ...] = ()
    reason_vi: str = ""

    @property
    def active(self) -> bool:
        return bool(self.categories)

    @property
    def reason_en(self) -> str:
        if self.mode == "manual":
            return (f"Manual: {len(self.categories)} folders ({', '.join(self.categories)})."
                    if self.active else "No folder restriction; searching the entire profile.")
        if self.mode == "auto":
            if not self.matches:
                return "No topic detected; searching the entire profile."
            if not self.active:
                return "Topic matching did not narrow the scope; searching the entire profile."
            labels = ", ".join(TOPIC_LABELS_EN.get(m.topic_id, m.topic_id) for m in self.matches)
            return f"Topic heuristic: {labels} (news folders and L30 remain in scope because they cover all topics)."
        return "Search all folders."

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "categories": list(self.categories),
            "strict_categories": list(self.strict_categories),
            "active": self.active,
            "reason_vi": self.reason_vi,
            "reason_en": self.reason_en,
            "matched_topics": [
                {
                    "topic_id": match.topic_id,
                    "label_vi": match.label_vi,
                    "label_en": TOPIC_LABELS_EN.get(match.topic_id, match.topic_id),
                    "keywords": list(match.keywords),
                    "categories": list(match.categories),
                }
                for match in self.matches
            ],
        }


def profile_categories(retrieval_database: str) -> tuple[str, ...]:
    """Categories one retrieval profile can return (empty for an unknown name)."""
    return PROFILE_CATEGORIES.get(retrieval_database, ())


def normalize_categories(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Upper-case, de-duplicate and drop blanks, preserving catalogue order."""
    wanted = {str(value or "").strip().upper() for value in (values or [])}
    wanted.discard("")
    ordered = [cat for cat in ALL_CATEGORIES if cat in wanted]
    # Anything outside the catalogue is kept at the end rather than silently
    # dropped, so a typo surfaces as an empty result instead of a full search.
    ordered += sorted(wanted - set(ordered))
    return tuple(ordered)


def _pattern(words: tuple[str, ...], *, fold: bool) -> re.Pattern[str] | None:
    """Word-boundary alternation over `words`, optionally diacritic-folded."""
    prepared = {(fold_vietnamese(word) if fold else word.lower()).strip() for word in words}
    prepared.discard("")
    if not prepared:
        return None
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in sorted(prepared, key=len, reverse=True)) + r")(?!\w)",
        re.UNICODE,
    )


_TOPIC_PATTERNS: dict[str, tuple[re.Pattern[str] | None, re.Pattern[str] | None]] = {
    topic.id: (_pattern(topic.keywords, fold=False), _pattern(topic.keywords, fold=True))
    for topic in TOPICS
}


def _is_unaccented(text: str) -> bool:
    """True when the operator typed Vietnamese without diacritics ("nau an")."""
    return fold_vietnamese(text) == text.lower()


def match_topics(text: str) -> tuple[TopicMatch, ...]:
    """Topics whose programme this query names.

    Matching is accent-sensitive whenever the query carries diacritics at all.
    Folding both sides looked more forgiving but silently equated distinct words:
    "vẫy tay đưa nghi phạm" folds onto the cycling term "tay đua", and "cụ lão"
    onto "cù lao". An operator typing without diacritics still gets the folded
    pass, because then there is nothing to be accent-sensitive about.
    """
    text = text or ""
    if not text.strip():
        return ()
    fold = _is_unaccented(text)
    haystack = fold_vietnamese(text) if fold else text.lower()
    matches: list[TopicMatch] = []
    for topic in TOPICS:
        pattern = _TOPIC_PATTERNS[topic.id][1 if fold else 0]
        hits = sorted(set(pattern.findall(haystack))) if pattern else []
        if hits:
            matches.append(
                TopicMatch(
                    topic_id=topic.id,
                    label_vi=topic.label_vi,
                    keywords=tuple(hits),
                    categories=topic.categories,
                )
            )
    return tuple(matches)


def suggest_categories(text: str, universe: tuple[str, ...]) -> ResolvedScope:
    """Route a query to the folders whose programme it is about."""
    matches = match_topics(text)
    if not matches:
        return ResolvedScope("auto", (), (), (), "Không có dấu hiệu chủ đề — tìm toàn bộ.")

    matched = normalize_categories([cat for match in matches for cat in match.categories])
    strict = tuple(cat for cat in matched if cat in universe)
    # Speciality cues narrow; the open-subject folders stay in scope regardless,
    # because that is where a topic can turn up outside its own programme.
    widened = normalize_categories([*matched, *OPEN_SUBJECT_CATEGORIES])
    categories = tuple(cat for cat in widened if cat in universe)

    if not categories or set(categories) >= set(universe):
        reason = (
            "Chủ đề trải khắp profile — tìm toàn bộ."
            if categories
            else "Chủ đề nhận ra không có thư mục nào trong profile này — tìm toàn bộ."
        )
        return ResolvedScope("auto", (), strict, matches, reason)

    labels = ", ".join(match.label_vi for match in matches)
    return ResolvedScope(
        "auto",
        categories,
        strict,
        matches,
        f"Heuristic chủ đề: {labels} (giữ lại thư mục thời sự + L30 vì các chương trình này đưa mọi chủ đề).",
    )


def resolve_scope(spec: dict | None, *, query: str, retrieval_database: str) -> ResolvedScope:
    """Turn a request's scope spec into the category set retrieval must obey.

    Modes: `all` (no filter), `manual` (exactly what the operator ticked), `auto`
    (topic heuristic, falling back to no filter). A manual selection covering the
    whole profile is normalized to no filter, so the adapters never build a
    filter that cannot exclude anything.
    """
    spec = spec or {}
    universe = profile_categories(retrieval_database)
    mode = str(spec.get("mode") or "all").lower()
    if mode not in {"all", "auto", "manual"}:
        mode = "all"

    if mode == "manual":
        categories = tuple(
            cat for cat in normalize_categories(spec.get("categories")) if cat in universe
        )
        if not categories:
            return ResolvedScope(
                "manual", (), (), (), "Chưa chọn thư mục hợp lệ nào — tìm toàn bộ."
            )
        if set(categories) >= set(universe):
            return ResolvedScope("manual", (), (), (), "Đã chọn toàn bộ thư mục.")
        return ResolvedScope(
            "manual",
            categories,
            categories,
            (),
            f"Thủ công: {len(categories)} thư mục ({', '.join(categories)}).",
        )

    if mode == "auto":
        return suggest_categories(query, universe)

    return ResolvedScope("all", (), (), (), "Tìm toàn bộ thư mục.")


# ---- push-down filters -------------------------------------------------
# `video_id` is a keyword field in every Elastic index and a VARCHAR in the
# Milvus image collection, and it always starts with its category plus a
# separator ("L21_", "N001-", see `identity.video_id_prefix`). Filtering on that
# prefix therefore works on every channel, unlike a category field that only the
# OCR and keyframe-map indices carry (speech and audio have none).

_SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9]{1,16}$")


def _safe(category: str) -> bool:
    """Only catalogue-shaped names reach a query string (no quotes, no wildcards)."""
    return bool(_SAFE_CATEGORY.match(category))


_SAFE_FAMILY = re.compile(r"^[A-Z][A-Za-z0-9]{0,7}-?$")
_SAFE_VIDEO_ID = re.compile(r"^[A-Z][A-Za-z0-9]{1,7}[-_]V\d{3}$")


@dataclass(frozen=True)
class FrameFilter:
    """Which frames of some video families may answer; frames of every other family may.

    A family is a video-id prefix: "N" is every traffic camera (no other folder's
    ids start with N), "S01-" the cycling race. Inside a constrained family only
    the listed videos pass, each whole (no intervals) or within its `keyframe_n`
    intervals. That is what lets a query-derived cue narrow the cameras without
    ever excluding a news bulletin: "Hai Bà Trưng" names a street camera and a
    historical figure alike, and only the camera side is filtered.

    `windows` is `((video_id, ((first_n, last_n), ...)), ...)`, sorted, so the
    filter is hashable and two equal filters build the same query text.
    """

    families: tuple[str, ...]
    windows: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]

    def __post_init__(self) -> None:
        if not self.families or not all(_SAFE_FAMILY.match(family) for family in self.families):
            raise ValueError(f"Invalid frame-filter families: {self.families!r}")
        for video_id, intervals in self.windows:
            if not _SAFE_VIDEO_ID.match(video_id) or not self._family_of(video_id):
                raise ValueError(f"Frame filter lists {video_id!r} outside {self.families!r}")
            if any(not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi) for lo, hi in intervals):
                raise ValueError(f"Invalid keyframe interval for {video_id}: {intervals!r}")

    def _family_of(self, video_id: str) -> str | None:
        return next((family for family in self.families if video_id.startswith(family)), None)

    @property
    def window_map(self) -> dict[str, tuple[tuple[int, int], ...]]:
        return dict(self.windows)

    def allows(self, video_id: str, keyframe_n: int) -> bool:
        if not self._family_of(video_id):
            return True
        intervals = self.window_map.get(video_id)
        if intervals is None:
            return False
        return not intervals or any(lo <= keyframe_n <= hi for lo, hi in intervals)

    def milvus_expr(self) -> str:
        """`(outside every family) or (an allowed video [inside its window])`."""
        outside = " and ".join(f'not (video_id like "{family}%")' for family in self.families)
        whole = [video_id for video_id, intervals in self.windows if not intervals]
        allowed = [f"video_id in [{', '.join(json.dumps(v) for v in whole)}]"] if whole else []
        for video_id, intervals in self.windows:
            if intervals:
                ranges = " or ".join(f"(keyframe_n >= {lo} and keyframe_n <= {hi})" for lo, hi in intervals)
                allowed.append(f'(video_id == "{video_id}" and ({ranges}))')
        return f"(({outside}) or {' or '.join(allowed)})" if allowed else f"({outside})"

    def elastic_clause(self) -> dict:
        allowed: list[dict] = [{"bool": {"must_not": [{"prefix": {"video_id": f}} for f in self.families]}}]
        whole = [video_id for video_id, intervals in self.windows if not intervals]
        if whole:
            allowed.append({"terms": {"video_id": whole}})
        for video_id, intervals in self.windows:
            if intervals:
                allowed.append({
                    "bool": {
                        "filter": [
                            {"term": {"video_id": video_id}},
                            {
                                "bool": {
                                    "should": [{"range": {"keyframe_n": {"gte": lo, "lte": hi}}} for lo, hi in intervals],
                                    "minimum_should_match": 1,
                                }
                            },
                        ]
                    }
                })
        return {"bool": {"should": allowed, "minimum_should_match": 1}}


def milvus_filter_expr(categories: tuple[str, ...], frames: FrameFilter | None = None) -> str:
    """Milvus boolean expression restricting a search to these categories (and frames)."""
    clauses = " or ".join(f'video_id like "{video_id_prefix(cat)}%"' for cat in categories if _safe(cat))
    parts = [f"({clauses})" if clauses else "", frames.milvus_expr() if frames else ""]
    return " and ".join(part for part in parts if part)


def elastic_filter_clause(categories: tuple[str, ...], frames: FrameFilter | None = None) -> dict | None:
    """Elastic bool clause restricting a search to these categories (and frames)."""
    prefixes = [{"prefix": {"video_id": video_id_prefix(cat)}} for cat in categories if _safe(cat)]
    clauses = [
        clause
        for clause in (
            {"bool": {"should": prefixes, "minimum_should_match": 1}} if prefixes else None,
            frames.elastic_clause() if frames else None,
        )
        if clause
    ]
    if len(clauses) > 1:
        return {"bool": {"filter": clauses}}
    return clauses[0] if clauses else None


def catalogue(retrieval_database: str) -> dict:
    """Everything the UI needs to draw the folder picker for one profile."""
    universe = profile_categories(retrieval_database)
    return {
        "retrieval_database": retrieval_database,
        "categories": [
            {
                "category": cat,
                "label_vi": CATEGORY_LABELS.get(cat, cat),
                "label_en": CATEGORY_LABELS_EN.get(cat, cat),
                "open_subject": cat in OPEN_SUBJECT_CATEGORIES,
            }
            for cat in universe
        ],
        "groups": [
            {"id": gid, "label_vi": label, "label_en": GROUP_LABELS_EN.get(gid, gid),
             "categories": [c for c in cats if c in universe], "collapsed": gid in COLLAPSED_GROUPS}
            for gid, label, cats in CATEGORY_GROUPS
            if any(c in universe for c in cats)
        ],
        "topics": [
            {
                "topic_id": topic.id,
                "label_vi": topic.label_vi,
                "label_en": TOPIC_LABELS_EN.get(topic.id, topic.id),
                "categories": [c for c in topic.categories if c in universe],
            }
            for topic in TOPICS
            if any(c in universe for c in topic.categories)
        ],
    }
