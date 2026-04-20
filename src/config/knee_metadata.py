"""
Centralized metadata for all knee videos: condition, cycles, and anatomical regions.

This module replaces scattered CYCLES, TYPES, and REGION_RANGES dictionaries.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
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

CSV_PATH = Path(__file__).resolve().parent / "metadata" / "knee_metadata.csv"


def _require_int(row: dict, col: str, ctx: str) -> int:
    v = row.get(col, "")
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{ctx}: expected int for column {col!r}, got {v!r}") from e


def _require_str(row: dict, col: str, ctx: str) -> str:
    v = row.get(col, "")
    if v is None:
        v = ""
    v = str(v).strip()
    if not v:
        raise ValueError(f"{ctx}: missing required column {col!r}")
    return v


def _load_knee_metadata_csv(csv_path: Path) -> Dict[Key, KneeVideoMeta]:
    if not csv_path.exists():
        raise FileNotFoundError(f"knee metadata CSV not found at {str(csv_path)!r}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"knee metadata CSV has no header: {str(csv_path)!r}")

        rows = list(reader)

    groups: dict[Key, list[dict]] = {}
    for i, row in enumerate(rows, start=2):
        ctx = f"{csv_path.name}:{i}"
        video_id = _require_int(row, "video_id", ctx)
        n_segments = _require_int(row, "n_segments", ctx)
        key = (video_id, n_segments)
        groups.setdefault(key, []).append(row)

    registry: Dict[Key, KneeVideoMeta] = {}
    errors: list[str] = []

    for (video_id, n_segments), group_rows in sorted(groups.items()):
        ctx = f"video_id={video_id}, n_segments={n_segments}"
        try:
            conditions = {_require_str(r, "condition", ctx) for r in group_rows}
            if len(conditions) != 1:
                raise ValueError(f"{ctx}: inconsistent condition values: {sorted(conditions)!r}")
            condition = next(iter(conditions))

            # Regions are repeated per row; dedupe and validate consistency.
            region_cols = [
                ("JC", "JC_start_1", "JC_end_1"),
                ("OT", "OT_start_1", "OT_end_1"),
                ("SB", "SB_start_1", "SB_end_1"),
            ]
            regions: Dict[str, RegionSegments] = {}
            for name, c_s, c_e in region_cols:
                starts = {_require_int(r, c_s, ctx) for r in group_rows}
                ends = {_require_int(r, c_e, ctx) for r in group_rows}
                if len(starts) != 1 or len(ends) != 1:
                    raise ValueError(
                        f"{ctx}: inconsistent region {name} values: {c_s}={sorted(starts)!r}, {c_e}={sorted(ends)!r}"
                    )
                regions[name] = RegionSegments(s=next(iter(starts)), e=next(iter(ends)))

            cycles_by_idx: dict[int, Cycle] = {}
            for r in group_rows:
                cycle_idx = _require_int(r, "cycle_idx", ctx)
                if cycle_idx in cycles_by_idx:
                    raise ValueError(f"{ctx}: duplicate cycle_idx={cycle_idx}")
                flex = FrameRange.from_1based(_require_int(r, "flex_start_1", ctx), _require_int(r, "flex_end_1", ctx))
                ext = FrameRange.from_1based(_require_int(r, "ext_start_1", ctx), _require_int(r, "ext_end_1", ctx))
                cycles_by_idx[cycle_idx] = Cycle(flex=flex, ext=ext)

            cycles: List[Cycle] = [c for _i, c in sorted(cycles_by_idx.items(), key=lambda kv: kv[0])]

            meta = KneeVideoMeta(
                condition=condition,
                video_id=video_id,
                n_segments=n_segments,
                cycles=cycles,
                regions=regions,
            )

            if (video_id, n_segments) in registry:
                raise ValueError(f"{ctx}: duplicate key encountered")
            registry[(video_id, n_segments)] = meta
        except Exception as e:
            errors.append(f"{ctx}: {e}")

    if errors:
        joined = "\n".join(f"- {m}" for m in errors)
        raise ValueError(f"knee metadata CSV validation failed ({csv_path}):\n{joined}")

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


KNEE_VIDEOS: Dict[Key, KneeVideoMeta] = _load_knee_metadata_csv(CSV_PATH)
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
