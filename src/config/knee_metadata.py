"""
Centralized interface for accessing metadata for all knee videos: condition, cycles, and anatomical regions.

This module replaces scattered CYCLES, TYPES, and REGION_RANGES dictionaries.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple, Literal, Iterable

Phase = Literal["flexion", "extension", "both"]

@dataclass(frozen=True)
class FrameRange:
    """0-based, inclusive frame range in video array indices."""
    s: int  # inclusive
    e: int    # inclusive

    def to_slice(self) -> slice:
        """For numpy slicing: video[frame_range.to_slice()]."""
        return slice(self.s, self.e + 1)

    @classmethod
    def from_1based(cls, start_1: int, end_1: int) -> "FrameRange":
        """Construct from 1-based inclusive frame numbers (e.g. Excel)."""
        return cls(s=start_1 - 1, e=end_1 - 1)


@dataclass(frozen=True)
class Cycle:
    """One flexion–extension cycle in absolute frame indices."""
    flex: FrameRange
    ext: FrameRange

    def full_cycle(self) -> FrameRange:
        """Overall range flex-start → ext-end."""
        return FrameRange(s=self.flex.s, e=self.ext.e)


@dataclass(frozen=True)
class RegionSegments:
    """Segment index range for anatomical regions.

    These stay 1-based to match notes/Excel; we expose a helper
    to convert to 0-based slice for arrays shaped (N_segments, ...).
    """
    s: int  # 1-based inclusive
    e: int    # 1-based inclusive

    def to_index_slice(self) -> slice:
        """0-based slice: total_sums[region_slice, :]."""
        return slice(self.s - 1, self.e)


@dataclass(frozen=True)
class KneeVideoMeta:
    """All metadata for a single knee video at a given N."""
    condition: str                 # e.g. "normal", "aging", "dmm-0w", "dmm-4w"
    video_id: int                  # e.g. 308, 1339, 1195
    n_segments: int                # e.g. 64

    # Ordered list of flexion–extension cycles
    cycles: List[Cycle]

    # Anatomical regions: JC, OT, SB
    regions: Dict[str, RegionSegments]  # keys: "JC", "OT", "SB"

    # (Optional) extra per-video metadata
    # metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cycle(self, idx: int) -> Cycle:
        return self.cycles[idx]

    def get_region(self, name: str) -> RegionSegments:
        return self.regions[name]


# Global registry: key = (video_id, n_segments)
Key = Tuple[int, int]  # (video_id, n_segments)

JSON_PATH = Path(__file__).resolve().parent / "metadata" / "knee_metadata.json"


def _require_int(obj: dict, key: str, ctx: str) -> int:
    v = obj.get(key, None)
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{ctx}: expected int for key {key!r}, got {v!r}") from e


def _require_str(obj: dict, key: str, ctx: str) -> str:
    v = obj.get(key, "")
    if v is None:
        v = ""
    v = str(v).strip()
    if not v:
        raise ValueError(f"{ctx}: missing required key {key!r}")
    return v


def _load_knee_metadata_json(json_path: Path) -> Dict[Key, KneeVideoMeta]:
    if not json_path.exists():
        raise FileNotFoundError(f"knee metadata JSON not found at {str(json_path)!r}")

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"failed to parse knee metadata JSON at {str(json_path)!r}") from e

    if not isinstance(raw, dict):
        raise ValueError(f"knee metadata JSON root must be an object; got {type(raw).__name__}")

    schema_version = raw.get("schema_version", None)
    if schema_version != 1:
        raise ValueError(f"unsupported knee metadata schema_version={schema_version!r}; expected 1")

    videos = raw.get("videos", None)
    if not isinstance(videos, list):
        raise ValueError(f"knee metadata JSON 'videos' must be a list; got {type(videos).__name__}")

    registry: Dict[Key, KneeVideoMeta] = {}
    errors: list[str] = []

    for i, entry in enumerate(videos):
        ctx = f"videos[{i}]"
        try:
            if not isinstance(entry, dict):
                raise ValueError(f"{ctx}: entry must be an object; got {type(entry).__name__}")

            video_id = _require_int(entry, "video_id", ctx)
            n_segments = _require_int(entry, "n_segments", ctx)
            condition = _require_str(entry, "condition", ctx)

            regions_raw = entry.get("regions", None)
            if not isinstance(regions_raw, dict):
                raise ValueError(f"{ctx}: 'regions' must be an object; got {type(regions_raw).__name__}")

            regions: Dict[str, RegionSegments] = {}
            for name in ("JC", "OT", "SB"):
                r_ctx = f"{ctx}.regions[{name!r}]"
                r_obj = regions_raw.get(name, None)
                if not isinstance(r_obj, dict):
                    raise ValueError(f"{r_ctx}: must be an object; got {type(r_obj).__name__}")
                s1 = _require_int(r_obj, "start_1", r_ctx)
                e1 = _require_int(r_obj, "end_1", r_ctx)
                regions[name] = RegionSegments(s=s1, e=e1)

            cycles_raw = entry.get("cycles", None)
            if not isinstance(cycles_raw, list) or not cycles_raw:
                raise ValueError(f"{ctx}: 'cycles' must be a non-empty list")

            cycles: List[Cycle] = []
            for j, c_obj in enumerate(cycles_raw):
                c_ctx = f"{ctx}.cycles[{j}]"
                if not isinstance(c_obj, dict):
                    raise ValueError(f"{c_ctx}: must be an object; got {type(c_obj).__name__}")

                flex_obj = c_obj.get("flex", None)
                ext_obj = c_obj.get("ext", None)
                if not isinstance(flex_obj, dict) or not isinstance(ext_obj, dict):
                    raise ValueError(f"{c_ctx}: 'flex' and 'ext' must be objects")

                flex = FrameRange.from_1based(
                    _require_int(flex_obj, "start_1", f"{c_ctx}.flex"),
                    _require_int(flex_obj, "end_1", f"{c_ctx}.flex"),
                )
                ext = FrameRange.from_1based(
                    _require_int(ext_obj, "start_1", f"{c_ctx}.ext"),
                    _require_int(ext_obj, "end_1", f"{c_ctx}.ext"),
                )
                cycles.append(Cycle(flex=flex, ext=ext))

            meta = KneeVideoMeta(
                condition=condition,
                video_id=video_id,
                n_segments=n_segments,
                cycles=cycles,
                regions=regions,
            )

            key = (video_id, n_segments)
            if key in registry:
                raise ValueError(f"{ctx}: duplicate key {key!r} encountered")
            registry[key] = meta
        except Exception as e:
            errors.append(f"{ctx}: {e}")

    if errors:
        joined = "\n".join(f"- {m}" for m in errors)
        raise ValueError(f"knee metadata JSON validation failed ({json_path}):\n{joined}")

    return registry


def validate_knee_metadata(registry: Dict[Key, KneeVideoMeta]) -> None:
    errors: list[str] = []

    for (video_id, n_segments), meta in sorted(registry.items()):
        ctx = f"video_id={video_id}, n_segments={n_segments}"

        # Regions: 1-based inclusive and cover [1, n_segments] contiguously.
        regions = meta.regions
        required = {"JC", "OT", "SB"}
        missing = required - set(regions.keys())
        if missing:
            errors.append(f"{ctx}: missing regions {sorted(missing)!r}")
            continue

        intervals = []
        for name in sorted(required):
            reg = regions[name]
            if not (1 <= reg.s <= reg.e <= n_segments):
                errors.append(
                    f"{ctx}: region {name} out of bounds or inverted: s={reg.s}, e={reg.e}, n_segments={n_segments}"
                )
            intervals.append((reg.s, reg.e, name))

        intervals.sort(key=lambda t: t[0])
        if intervals:
            if intervals[0][0] != 1:
                errors.append(f"{ctx}: regions must start at segment 1; got start={intervals[0][0]} ({intervals[0][2]})")
            for (ps, pe, pn), (ns, ne, nn) in zip(intervals, intervals[1:]):
                if ns != pe + 1:
                    errors.append(
                        f"{ctx}: regions not contiguous between {pn}({ps}-{pe}) and {nn}({ns}-{ne}); expected next.start={pe+1}"
                    )
            if intervals[-1][1] != n_segments:
                errors.append(
                    f"{ctx}: regions must end at n_segments={n_segments}; got end={intervals[-1][1]} ({intervals[-1][2]})"
                )

        # Cycles: validate ordering and bounds (FrameRange is already 0-based inclusive).
        if not meta.cycles:
            errors.append(f"{ctx}: no cycles defined")
        for i, c in enumerate(meta.cycles):
            if c.flex.s < 0 or c.flex.e < 0 or c.ext.s < 0 or c.ext.e < 0:
                errors.append(f"{ctx}: cycle_idx={i}: negative frame index in {c}")
                continue
            if c.flex.s > c.flex.e:
                errors.append(f"{ctx}: cycle_idx={i}: flex start > end ({c.flex.s}>{c.flex.e})")
            if c.ext.s > c.ext.e:
                errors.append(f"{ctx}: cycle_idx={i}: ext start > end ({c.ext.s}>{c.ext.e})")
            # Allow touching boundary (<=) due to existing data (e.g. normal/1193).
            if c.flex.e > c.ext.s:
                errors.append(
                    f"{ctx}: cycle_idx={i}: flex must end before or at ext start; flex_end={c.flex.e}, ext_start={c.ext.s}"
                )

    if errors:
        joined = "\n".join(f"- {m}" for m in errors)
        raise ValueError(f"knee metadata validation failed:\n{joined}")


KNEE_VIDEOS: Dict[Key, KneeVideoMeta] = _load_knee_metadata_json(JSON_PATH)
validate_knee_metadata(KNEE_VIDEOS)


def get_knee_meta(video_id: int, n_segments: int) -> KneeVideoMeta:
    key = (int(video_id), int(n_segments))
    try:
        return KNEE_VIDEOS[key]
    except KeyError:
        raise KeyError(f"No metadata for {key=}")


def get_knee_meta_by_condition(condition: str, video_id: int, n_segments: int) -> KneeVideoMeta:
    """Compatibility wrapper for older call sites.

    The registry is keyed by (video_id, n_segments); this asserts the passed
    condition matches the CSV to catch mismatched wiring early.
    """
    meta = get_knee_meta(video_id, n_segments)
    if str(condition) != meta.condition:
        raise KeyError(
            f"No metadata for condition={condition!r}, video_id={int(video_id)}, n_segments={int(n_segments)}; "
            f"found condition={meta.condition!r} in CSV"
        )
    return meta
