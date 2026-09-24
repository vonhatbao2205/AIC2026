"""Batch-2 keyframe registry contract (M news, N traffic cameras, S01 cycling).

Shared by the two batch-2 uploaders:

* ``elastic_upload_batch2_keyframe_map.py`` — keyframe map → Elastic
* ``milvus_upload_pe_core_batch2.py``       — PE-Core-G14-448 vectors → Milvus

The source of truth is the three audited ``final/frame_registry.parquet`` files
(``Batch2/AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`` §5.1, §12.1), one row per
JPEG on R2. Each is pinned by SHA-256, so an uploader can never mix rows from a
re-run keyframe snapshot with vectors encoded from the old one.

Identity differs from L in two ways, and both are load-bearing:

* the JPEG on R2 is named by the ordinal ``n`` (``001.jpg``), not by
  ``frame_idx``; the application id is still ``<category>/<video_id>/<n:03d>``;
* N and S01 video ids use a hyphen (``N001-V001``, ``S01-V001``), M an
  underscore (``M01_V001``). The category is therefore never ``split("_")``
  of the video id — it is matched per profile below.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REGISTRY_ROOT = Path("/home/bao/Projects/ExtractKeyframe/keyframe_batch2")
R2_BUCKET = "aic26-infoshot-keyframes"
#: Frames below this were not embedded: black tail of M10_V029 and fades (handoff §11.1–11.2).
MIN_QUALITY = 0.05
REGISTRY_COLUMNS = (
    "n",
    "frame_id",
    "video_id",
    "category",
    "frame_idx",
    "pts_time",
    "fps",
    "quality",
    "keyframe_name",
    "submit_keyframe_id",
    "r2_bucket",
    "r2_key",
    "pipeline_signature",
)


@dataclass(frozen=True)
class Profile:
    name: str
    directory: str
    pipeline_signature: str
    registry_sha256: str
    rows: int
    videos: int
    categories: int
    #: Full match of a video id; group 1 is its category.
    video_pattern: re.Pattern[str]

    @property
    def registry_relpath(self) -> str:
        return f"{self.directory}/final/frame_registry.parquet"


PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (
        Profile(
            name="M",
            directory="infoshoot_m",
            pipeline_signature="e6c789322adb",
            registry_sha256="7b34dc36a3e69ae514e5d473a7c77645e71790a6ab24e47a486a20d8d8f460ae",
            rows=209_111,
            videos=304,
            categories=10,
            video_pattern=re.compile(r"(M(?:0[1-9]|10))_V\d{3}"),
        ),
        Profile(
            name="N",
            directory="infoshoot_n",
            pipeline_signature="773e4b7465c2",
            registry_sha256="523b1a11cadc0567233d47f7a46a5012eb1c8dc57df95ab0e00f40ac4d6b89fa",
            rows=44_572,
            videos=298,
            categories=100,
            video_pattern=re.compile(r"(N\d{3})-V\d{3}"),
        ),
        Profile(
            name="S",
            directory="infoshoot_s",
            pipeline_signature="db5918a110cf",
            registry_sha256="b3d7c630a2e71f04a445fbe6ab5020568ce098fca70fc34f911f20bc5a024a54",
            rows=621_117,
            videos=12,
            categories=1,
            video_pattern=re.compile(r"(S01)-V\d{3}"),
        ),
    )
}
TOTAL_KEYFRAMES = sum(profile.rows for profile in PROFILES.values())  # 874,800


@dataclass(frozen=True, slots=True)
class KeyframeRow:
    profile: str
    frame_id: str
    video_id: str
    category: str
    n: int
    frame_idx: int
    pts_time: float
    fps: float
    quality: float
    r2_key: str

    @property
    def keyframe_name(self) -> str:
        return f"{self.n:03d}"

    @property
    def keyframe_id(self) -> str:
        return f"{self.video_id}/{self.keyframe_name}"

    @property
    def submit_keyframe_id(self) -> str:
        return f"{self.category}/{self.keyframe_id}"

    @property
    def embedded(self) -> bool:
        return self.quality >= MIN_QUALITY


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def profile_for_video(video_id: str) -> tuple[Profile, str]:
    """The profile and category of a batch-2 video id, or ValueError."""
    for profile in PROFILES.values():
        match = profile.video_pattern.fullmatch(video_id)
        if match:
            return profile, match.group(1)
    raise ValueError(f"Not a batch-2 video id: {video_id!r}")


def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def validate_rows(profile: Profile, columns: dict[str, list]) -> list[KeyframeRow]:
    """Check every row against the identity contract; return rows ordered by (video, n)."""
    count = len(columns["frame_id"])
    rows: list[KeyframeRow] = []
    seen_frame_ids: set[str] = set()
    for index in range(count):
        value = {name: columns[name][index] for name in REGISTRY_COLUMNS}
        frame_id = str(value["frame_id"])
        video_id = str(value["video_id"])
        where = f"{profile.registry_relpath} row {index} ({frame_id})"
        matched = profile.video_pattern.fullmatch(video_id)
        if not matched:
            raise ValueError(f"{where}: video_id {video_id!r} does not belong to profile {profile.name}")
        category = matched.group(1)
        n = int(value["n"])
        frame_idx = int(value["frame_idx"])
        pts_time = float(value["pts_time"])
        fps = float(value["fps"])
        quality = float(value["quality"])
        expected = {
            "category": category,
            "frame_id": f"{video_id}@f{frame_idx:08d}",
            "keyframe_name": f"{n:03d}.jpg",
            "submit_keyframe_id": f"{category}/{video_id}/{n:03d}",
            "r2_bucket": R2_BUCKET,
            "r2_key": f"Keyframes/Keyframes_{category}/{video_id}/{n:03d}.jpg",
            "pipeline_signature": profile.pipeline_signature,
        }
        for name, wanted in expected.items():
            if value[name] != wanted:
                raise ValueError(f"{where}: {name}={value[name]!r}, expected {wanted!r}")
        if n < 1 or frame_idx < 0:
            raise ValueError(f"{where}: invalid n={n} / frame_idx={frame_idx}")
        if not (_finite(pts_time) and _finite(fps) and fps > 0 and _finite(quality)):
            raise ValueError(f"{where}: invalid pts_time/fps/quality")
        if frame_id in seen_frame_ids:
            raise ValueError(f"{where}: duplicate frame_id")
        seen_frame_ids.add(frame_id)
        rows.append(
            KeyframeRow(
                profile=profile.name,
                frame_id=frame_id,
                video_id=video_id,
                category=category,
                n=n,
                frame_idx=frame_idx,
                pts_time=pts_time,
                fps=fps,
                quality=quality,
                r2_key=value["r2_key"],
            )
        )

    rows.sort(key=lambda row: (row.video_id, row.n))
    previous: KeyframeRow | None = None
    for row in rows:
        if previous is None or previous.video_id != row.video_id:
            if row.n != 1:
                raise ValueError(f"{row.video_id}: first keyframe is n={row.n}, expected 1")
        else:
            if row.n != previous.n + 1:
                raise ValueError(f"{row.video_id}: n jumps {previous.n} -> {row.n}")
            if row.frame_idx <= previous.frame_idx:
                raise ValueError(f"{row.video_id}: frame_idx not strictly increasing at n={row.n}")
        previous = row
    return rows


def load_registry(profile: Profile, root: Path = DEFAULT_REGISTRY_ROOT) -> list[KeyframeRow]:
    """Read one pinned registry and prove it is exactly the audited snapshot."""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing pyarrow. Install it with: uv pip install pyarrow") from exc

    path = root / profile.registry_relpath
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != profile.registry_sha256:
        raise ValueError(
            f"{path}: SHA-256 {digest[:12]}… differs from the audited snapshot "
            f"{profile.registry_sha256[:12]}… — the keyframe snapshot changed; do not upload."
        )
    table = pq.read_table(path, columns=list(REGISTRY_COLUMNS))
    rows = validate_rows(profile, {name: table.column(name).to_pylist() for name in REGISTRY_COLUMNS})
    videos = {row.video_id for row in rows}
    categories = {row.category for row in rows}
    if len(rows) != profile.rows or len(videos) != profile.videos or len(categories) != profile.categories:
        raise ValueError(
            f"{path}: {len(rows):,} rows / {len(videos)} videos / {len(categories)} categories, expected "
            f"{profile.rows:,} / {profile.videos} / {profile.categories}"
        )
    return rows


def load_registries(
    names: tuple[str, ...] = tuple(PROFILES), root: Path = DEFAULT_REGISTRY_ROOT
) -> dict[str, list[KeyframeRow]]:
    result: dict[str, list[KeyframeRow]] = {}
    for name in names:
        rows = load_registry(PROFILES[name], root)
        embedded = sum(row.embedded for row in rows)
        print(
            f"[registry {name}] {len(rows):,} keyframes, {len({row.video_id for row in rows})} videos, "
            f"{embedded:,} at quality ≥ {MIN_QUALITY} — SHA-256 pinned, identity contract PASS",
            flush=True,
        )
        result[name] = rows
    return result
