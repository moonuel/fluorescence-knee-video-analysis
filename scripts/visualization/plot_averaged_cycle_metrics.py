"""
Multi-video cycle-averaged metrics from intensity Excel workbooks.

This script replaces `plot_com_cycles_from_heatmaps.py` with a CLI more like
`dmm_analysis.py`, using per-video Excel workbooks in `data/intensities_total/`.

Usage:
    python scripts/visualization/plot_averaged_cycle_metrics.py \
        VIDEO_SPEC [VIDEO_SPEC ...] \
        --metric {com,total,flux} \
        --phase {flexion,extension,both} \
        --scaling {raw,norm,rel} \
        --source {raw,bgsub} \
        [--x-domain {angle,frame}] [--n-interp-samples N] \
        [--out-dir PATH] [--save] [--no-show] [--export-xlsx]

VIDEO_SPEC grammar:
    VIDEO_SPEC := BASE [ ":" KVPAIR ]...
    BASE       := <video_id>"N"<n_segments> | <workbook_basename.xlsx>
    KVPAIR     := "cycles=" CYCLES | "label=" LABEL
    CYCLES     := "all" | <int>[,<int>...]

Examples:
    python scripts/visualization/plot_averaged_cycle_metrics.py \
        1339N64:cycles=1,2,3 \
        1342N64:cycles=2 \
        --metric com --phase both --scaling norm --source raw

    python scripts/visualization/plot_averaged_cycle_metrics.py \
        1190N64:cycles=all \
        1207N64:cycles=1,2 \
        --metric flux --phase both --scaling rel --x-domain angle
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

MetricType = Literal["com", "total", "flux"]
PhaseType = Literal["flexion", "extension", "both"]
ScalingType = Literal["raw", "norm", "rel"]
SourceType = Literal["raw", "bgsub"]
XDomainType = Literal["angle", "frame"]

# Sheet names
SHEET_SEGMENT_INTENSITIES = "Segment Intensities"
SHEET_SEGMENT_INTENSITIES_BGSUB = "Segment Intensities BGSub"
SHEET_FLEXION_FRAMES = "Flexion Frames"
SHEET_EXTENSION_FRAMES = "Extension Frames"
SHEET_ANATOMICAL_REGIONS = "Anatomical Regions"

# Default directories
INTENSITIES_DIR = Path("data") / "intensities_total"
DEFAULT_OUT_DIR = Path("figures") / "cycle_metrics_from_intensities"


# =============================================================================
# VIDEO_SPEC PARSING
# =============================================================================

# Regex for shorthand BASE: VIDNSEGS (e.g., 1339N64, 0308N64)
_RE_SHORTHAND = re.compile(r"^(\d+)N(\d+)$", re.IGNORECASE)

# Regex for Excel basename
_RE_EXCEL_BASENAME = re.compile(r"^.+\.xlsx$", re.IGNORECASE)


@dataclass
class VideoSpec:
    """Parsed VIDEO_SPEC representation.

    Attributes:
        base: The BASE portion (shorthand like '1339N64' or basename like '1339N64intensities.xlsx').
        video_id: Extracted video ID (int).
        n_segments: Extracted segment count (int).
        cycles: List of 1-based cycle indices, or None for 'all'.
        label: Optional custom label for legend.
        resolved_path: Resolved path to the Excel workbook (set after resolution).
    """
    base: str
    video_id: Optional[int] = None
    n_segments: Optional[int] = None
    cycles: Optional[List[int]] = None  # None means 'all'
    label: Optional[str] = None
    resolved_path: Optional[Path] = None


def parse_video_spec(spec_str: str) -> VideoSpec:
    """Parse a single VIDEO_SPEC string.

    Args:
        spec_str: A VIDEO_SPEC string (e.g., '1339N64:cycles=1,2,3:label=aging').

    Returns:
        VideoSpec with parsed components.

    Raises:
        ValueError: If the spec is malformed.
    """
    parts = spec_str.split(":")
    base = parts[0].strip()

    if not base:
        raise ValueError(f"VIDEO_SPEC has empty BASE: {spec_str!r}")

    # Reject paths with directory separators
    if os.path.sep in base or "/" in base or "\\" in base:
        raise ValueError(
            f"VIDEO_SPEC BASE must be a basename, not a path: {base!r}. "
            f"Inputs are resolved under {INTENSITIES_DIR}/"
        )

    # Initialize spec
    spec = VideoSpec(base=base)

    # Try to extract video_id and n_segments from shorthand
    m_shorthand = _RE_SHORTHAND.match(base)
    if m_shorthand:
        spec.video_id = int(m_shorthand.group(1))
        spec.n_segments = int(m_shorthand.group(2))
    elif _RE_EXCEL_BASENAME.match(base):
        # Try to extract from basename like '1339N64intensities.xlsx'
        # Look for pattern <digits>N<digits> in the basename
        m_embedded = re.search(r"(\d+)N(\d+)", base, re.IGNORECASE)
        if m_embedded:
            spec.video_id = int(m_embedded.group(1))
            spec.n_segments = int(m_embedded.group(2))
    else:
        raise ValueError(
            f"VIDEO_SPEC BASE must be shorthand (e.g., '1339N64') or an Excel basename "
            f"(e.g., '1339N64intensities.xlsx'): {base!r}"
        )

    # Parse key-value pairs
    for kv_part in parts[1:]:
        kv_part = kv_part.strip()
        if not kv_part:
            continue

        if "=" not in kv_part:
            raise ValueError(
                f"VIDEO_SPEC KVPAIR must contain '=': {kv_part!r} in {spec_str!r}"
            )

        key, value = kv_part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "cycles":
            if value.lower() == "all":
                spec.cycles = None  # 'all' is represented as None
            else:
                try:
                    cycle_indices = [int(x.strip()) for x in value.split(",") if x.strip()]
                except ValueError as e:
                    raise ValueError(
                        f"VIDEO_SPEC cycles must be 'all' or comma-separated integers: {value!r}"
                    ) from e

                if any(idx < 1 for idx in cycle_indices):
                    raise ValueError(
                        f"VIDEO_SPEC cycle indices must be >= 1: {cycle_indices}"
                    )

                spec.cycles = cycle_indices

        elif key == "label":
            spec.label = value

        else:
            raise ValueError(
                f"VIDEO_SPEC unknown KVPAIR key: {key!r}. "
                f"Supported keys: 'cycles', 'label'"
            )

    return spec


def resolve_video_spec_path(spec: VideoSpec) -> Path:
    """Resolve a VideoSpec to an actual file path.

    Args:
        spec: VideoSpec with base set.

    Returns:
        Resolved Path to the Excel workbook.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the BASE is ambiguous (multiple matches).
    """
    candidates: List[Path] = []

    # Check if base is already an Excel basename
    if _RE_EXCEL_BASENAME.match(spec.base):
        direct_path = INTENSITIES_DIR / spec.base
        if direct_path.is_file():
            candidates.append(direct_path)
    else:
        # Shorthand: try both with and without leading zeros
        # e.g., 308N64 -> 308N64intensities.xlsx or 0308N64intensities.xlsx
        pattern = f"{spec.base}intensities.xlsx"
        direct_path = INTENSITIES_DIR / pattern
        if direct_path.is_file():
            candidates.append(direct_path)

        # Also check for zero-padded variants
        if spec.video_id is not None:
            # Try zero-padded to 4 digits
            padded_pattern = f"{spec.video_id:04d}N{spec.n_segments}intensities.xlsx"
            padded_path = INTENSITIES_DIR / padded_pattern
            if padded_path.is_file() and padded_path not in candidates:
                candidates.append(padded_path)

            # Try without zero padding
            unpadded_pattern = f"{spec.video_id}N{spec.n_segments}intensities.xlsx"
            unpadded_path = INTENSITIES_DIR / unpadded_pattern
            if unpadded_path.is_file() and unpadded_path not in candidates:
                candidates.append(unpadded_path)

    if not candidates:
        raise FileNotFoundError(
            f"No workbook found for VIDEO_SPEC {spec.base!r} under {INTENSITIES_DIR}/"
        )

    if len(candidates) > 1:
        # Ambiguous: user must specify exact basename
        candidate_names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Ambiguous BASE {spec.base!r}. Multiple matches: {candidate_names}. "
            f"Specify the exact basename (e.g., '0308N64intensities.xlsx')."
        )

    return candidates[0]


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-video cycle-averaged metrics from intensity Excel workbooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VIDEO_SPEC examples:
  1339N64:cycles=1,2,3     Video 1339, 64 segments, cycles 1-3
  1342N64:cycles=2         Video 1342, 64 segments, cycle 2 only
  308N64                   Video 308, 64 segments, all cycles (default)
  0308N64intensities.xlsx:cycles=1,2   Explicit basename
  1339N64:cycles=all:label=aging-1339  Custom legend label
        """,
    )

    # Positional: VIDEO_SPEC(s) - optional if --list is used
    parser.add_argument(
        "video_specs",
        nargs="*",
        metavar="VIDEO_SPEC",
        help="One or more video specifications (see examples below). Required unless --list is used.",
    )

    # Metric selection
    parser.add_argument(
        "--metric",
        choices=["com", "total", "flux"],
        default="total",
        help="Metric to compute: com (center-of-mass), total (sum intensity), flux (boundary flux). "
             "Exactly one metric per invocation. (default: total)",
    )

    # Phase selection
    parser.add_argument(
        "--phase",
        choices=["flexion", "extension", "both"],
        default="both",
        help="Phase to plot (default: both)",
    )

    # Scaling
    parser.add_argument(
        "--scaling",
        choices=["raw", "norm", "rel"],
        default="raw",
        help="Intensity scaling mode: raw (no scaling), norm (per-frame min-max to 0-100), "
             "rel (per-frame relative intensity). (default: raw)",
    )

    # Source
    parser.add_argument(
        "--source",
        choices=["raw", "bgsub"],
        default="raw",
        help="Data source: raw (Segment Intensities sheet), bgsub (Segment Intensities BGSub sheet). "
             "(default: raw)",
    )

    # X-axis domain
    parser.add_argument(
        "--x-domain",
        choices=["angle", "frame"],
        default="angle",
        help="X-axis domain: angle (interpolate to angle grid), frame (centered relative frames). "
             "(default: angle)",
    )

    parser.add_argument(
        "--n-interp-samples",
        type=int,
        default=525,
        help="Number of interpolation samples per phase when --x-domain angle. (default: 525)",
    )

    # Output control
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workbook basenames under data/intensities_total/",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures. (default: {DEFAULT_OUT_DIR})",
    )

    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=True,
        help="Save figures as PDF (default: enabled)",
    )

    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save figures",
    )

    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display figures interactively (default: enabled)",
    )

    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display figures interactively",
    )

    parser.add_argument(
        "--export-xlsx",
        action="store_true",
        default=False,
        help="Export averaged intensities and metrics to Excel",
    )

    return parser


@dataclass
class CliArgs:
    """Parsed CLI arguments."""
    video_specs: List[VideoSpec]
    metric: MetricType
    phase: PhaseType
    scaling: ScalingType
    source: SourceType
    x_domain: XDomainType
    n_interp_samples: int
    list_mode: bool
    out_dir: Path
    save: bool
    show: bool
    export_xlsx: bool


def parse_args(argv: Optional[List[str]] = None) -> CliArgs:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        CliArgs with parsed values.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # If --list is set, we don't need VIDEO_SPEC
    if args.list:
        return CliArgs(
            video_specs=[],
            metric=args.metric,
            phase=args.phase,
            scaling=args.scaling,
            source=args.source,
            x_domain=args.x_domain,
            n_interp_samples=args.n_interp_samples,
            list_mode=True,
            out_dir=args.out_dir,
            save=args.save,
            show=args.show,
            export_xlsx=args.export_xlsx,
        )

    # Otherwise, VIDEO_SPEC is required
    if not args.video_specs:
        parser.error("VIDEO_SPEC is required (unless --list is used)")

    # Parse VIDEO_SPEC strings
    parsed_specs: List[VideoSpec] = []
    for spec_str in args.video_specs:
        try:
            spec = parse_video_spec(spec_str)
            parsed_specs.append(spec)
        except ValueError as e:
            parser.error(f"Invalid VIDEO_SPEC {spec_str!r}: {e}")

    return CliArgs(
        video_specs=parsed_specs,
        metric=args.metric,
        phase=args.phase,
        scaling=args.scaling,
        source=args.source,
        x_domain=args.x_domain,
        n_interp_samples=args.n_interp_samples,
        list_mode=args.list,
        out_dir=args.out_dir,
        save=args.save,
        show=args.show,
        export_xlsx=args.export_xlsx,
    )


# =============================================================================
# DATA LOADING STUBS
# =============================================================================

@dataclass
class WorkbookData:
    """Loaded data from an intensity Excel workbook.

    Attributes:
        path: Path to the workbook.
        video_id: Video identifier.
        n_segments: Number of segments.
        intensities: Per-segment intensity matrix (n_frames, n_segments).
        flexion_cycles: Flexion cycle frame ranges.
        extension_cycles: Extension cycle frame ranges.
        anatomical_regions: Anatomical region definitions.
    """
    path: Path
    video_id: int
    n_segments: int
    intensities: np.ndarray
    flexion_cycles: pd.DataFrame
    extension_cycles: pd.DataFrame
    anatomical_regions: pd.DataFrame


def load_workbook(spec: VideoSpec, source: SourceType) -> WorkbookData:
    """Load data from an intensity Excel workbook.

    Args:
        spec: Resolved VideoSpec.
        source: Data source ('raw' or 'bgsub').

    Returns:
        WorkbookData with loaded arrays.

    Raises:
        FileNotFoundError: If required sheets are missing.
        ValueError: If data is malformed.
    """
    def _require_columns(df: pd.DataFrame, cols: List[str], sheet: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Malformed sheet {sheet!r} in workbook {path}: missing columns {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    def _clean_cycle_sheet(df_raw: pd.DataFrame, sheet: str) -> pd.DataFrame:
        """Return DataFrame with integer columns ['start','end'].

        Handles legacy formats where the sheet may have:
        - a header row with 'Start'/'End'
        - three columns: cycle/index, start, end
        - extra blank rows/columns
        """
        if df_raw.empty:
            raise ValueError(f"Sheet {sheet!r} in workbook {path} is empty")

        df = df_raw.copy()
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError(f"Sheet {sheet!r} in workbook {path} has no data")

        # If the first row looks like a header, drop it.
        first_row = df.iloc[0, :].astype(str).str.lower()
        if ("start" in first_row.values) and ("end" in first_row.values):
            df = df.iloc[1:, :]

        # Keep first 3 columns max: [cycle?], start, end
        df = df.iloc[:, 0:3]

        # Coerce start/end from 2nd/3rd columns.
        start = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        end = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        out = pd.DataFrame({"start": start, "end": end}).dropna(how="any")
        out["start"] = out["start"].astype(int)
        out["end"] = out["end"].astype(int)

        if out.empty:
            raise ValueError(
                f"Sheet {sheet!r} in workbook {path} has no valid (start,end) rows after cleaning"
            )
        return out.reset_index(drop=True)

    # Resolve workbook path
    path = spec.resolved_path or resolve_video_spec_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")

    intensity_sheet = (
        SHEET_SEGMENT_INTENSITIES if source == "raw" else SHEET_SEGMENT_INTENSITIES_BGSUB
    )

    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        raise ValueError(f"Failed to open workbook {path}: {e}") from e

    required_sheets = [
        intensity_sheet,
        SHEET_FLEXION_FRAMES,
        SHEET_EXTENSION_FRAMES,
        SHEET_ANATOMICAL_REGIONS,
    ]
    missing_sheets = [s for s in required_sheets if s not in xls.sheet_names]
    if missing_sheets:
        raise FileNotFoundError(
            f"Workbook {path} is missing required sheets: {missing_sheets}. "
            f"Available: {xls.sheet_names}"
        )

    # Intensities: written by prepare_intensity_data.py via DataFrame.to_excel(index=True)
    # so we expect headers in row 1 and an index column.
    df_int = pd.read_excel(xls, sheet_name=intensity_sheet, header=0, index_col=0)
    # Some legacy workbooks may include non-numeric columns; coerce and fill.
    df_int = df_int.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    intensities = df_int.to_numpy(dtype=float)

    # Cycles: read raw then clean to (start,end)
    df_flex_raw = pd.read_excel(xls, sheet_name=SHEET_FLEXION_FRAMES, header=None)
    df_ext_raw = pd.read_excel(xls, sheet_name=SHEET_EXTENSION_FRAMES, header=None)
    flexion_cycles = _clean_cycle_sheet(df_flex_raw, SHEET_FLEXION_FRAMES)
    extension_cycles = _clean_cycle_sheet(df_ext_raw, SHEET_EXTENSION_FRAMES)

    # Anatomical regions: expected columns Region, Start, End
    anatomical_regions = pd.read_excel(xls, sheet_name=SHEET_ANATOMICAL_REGIONS, header=0)
    anatomical_regions.columns = [str(c).strip() for c in anatomical_regions.columns]
    # Normalize likely variants.
    colmap: Dict[str, str] = {}
    for c in anatomical_regions.columns:
        cl = c.lower()
        if cl == "region":
            colmap[c] = "Region"
        elif cl == "start":
            colmap[c] = "Start"
        elif cl == "end":
            colmap[c] = "End"
    anatomical_regions = anatomical_regions.rename(columns=colmap)
    _require_columns(anatomical_regions, ["Region", "Start", "End"], SHEET_ANATOMICAL_REGIONS)
    anatomical_regions = anatomical_regions.dropna(subset=["Region", "Start", "End"]).copy()
    anatomical_regions["Start"] = pd.to_numeric(anatomical_regions["Start"], errors="coerce")
    anatomical_regions["End"] = pd.to_numeric(anatomical_regions["End"], errors="coerce")
    anatomical_regions = anatomical_regions.dropna(subset=["Start", "End"]).copy()
    anatomical_regions["Start"] = anatomical_regions["Start"].astype(int)
    anatomical_regions["End"] = anatomical_regions["End"].astype(int)

    # Infer identifiers / validate n_segments
    n_segments = int(df_int.shape[1])
    if spec.n_segments is not None and int(spec.n_segments) != n_segments:
        raise ValueError(
            f"Segment count mismatch for {path.name}: spec has N={spec.n_segments}, "
            f"but sheet {intensity_sheet!r} has {n_segments} columns"
        )

    video_id = spec.video_id
    if video_id is None:
        m = re.search(r"(\d+)N(\d+)", path.name, re.IGNORECASE)
        if m:
            video_id = int(m.group(1))
    if video_id is None:
        raise ValueError(
            f"Could not infer video_id from VIDEO_SPEC {spec.base!r} or workbook name {path.name!r}"
        )

    return WorkbookData(
        path=path,
        video_id=int(video_id),
        n_segments=n_segments,
        intensities=intensities,
        flexion_cycles=flexion_cycles,
        extension_cycles=extension_cycles,
        anatomical_regions=anatomical_regions.reset_index(drop=True),
    )


def validate_cycles(spec: VideoSpec, workbook: WorkbookData, phase: PhaseType) -> None:
    """Validate that requested cycle indices are in range and frame intervals are valid.

    Cycle indices are interpreted as 1-based (matching the CLI and typical worksheet labeling).
    Frame intervals are expected to be 1-based inclusive, as written by
    scripts/analysis/prepare_intensity_data.py.

    Args:
        spec: VideoSpec with cycles to validate.
        workbook: Loaded workbook data.
        phase: Phase selection controlling which cycle tables are required.

    Raises:
        ValueError: If any requested cycle index is out of range for the required phase(s),
            or if any referenced start/end frame is out of bounds.
    """
    if spec.cycles is None:
        return

    n_flex = int(len(workbook.flexion_cycles))
    n_ext = int(len(workbook.extension_cycles))
    n_frames = int(workbook.intensities.shape[0])

    if phase == "flexion":
        max_cycle = n_flex
    elif phase == "extension":
        max_cycle = n_ext
    else:
        # For paired plots/metrics, require that a cycle exists in both phases.
        max_cycle = min(n_flex, n_ext)

    if max_cycle <= 0:
        raise ValueError(
            f"No cycles available for {spec.base!r} (phase={phase}). "
            f"flexion_cycles={n_flex}, extension_cycles={n_ext}"
        )

    bad = [c for c in spec.cycles if c < 1 or c > max_cycle]
    if bad:
        raise ValueError(
            f"Requested cycle indices out of range for {spec.base!r} (phase={phase}). "
            f"Requested={spec.cycles}, valid=1..{max_cycle}. "
            f"(flexion_cycles={n_flex}, extension_cycles={n_ext})"
        )

    def _check_intervals(df: pd.DataFrame, sheet_name: str, cyc_idxs: List[int]) -> None:
        for cyc in cyc_idxs:
            row = df.iloc[cyc - 1]
            try:
                start = int(row["start"])
                end = int(row["end"])
            except Exception as e:
                raise ValueError(
                    f"Malformed cycle row for {spec.base!r} in sheet {sheet_name!r}: "
                    f"cycle={cyc}, row={row.to_dict()}"
                ) from e

            if start < 1 or end < 1:
                raise ValueError(
                    f"Invalid (start,end) in {sheet_name!r} for {spec.base!r}: "
                    f"cycle={cyc}, start={start}, end={end} (must be >= 1)"
                )
            if start > end:
                raise ValueError(
                    f"Invalid (start,end) in {sheet_name!r} for {spec.base!r}: "
                    f"cycle={cyc}, start={start} > end={end}"
                )
            if end > n_frames:
                raise ValueError(
                    f"Cycle frames out of bounds for {spec.base!r}: sheet={sheet_name!r}, "
                    f"cycle={cyc}, end={end} exceeds n_frames={n_frames}"
                )

    if phase in ("flexion", "both"):
        _check_intervals(workbook.flexion_cycles, SHEET_FLEXION_FRAMES, spec.cycles)
    if phase in ("extension", "both"):
        _check_intervals(workbook.extension_cycles, SHEET_EXTENSION_FRAMES, spec.cycles)


# =============================================================================
# INTENSITY SCALING STUBS
# =============================================================================

def apply_intensity_scaling(intensities: np.ndarray, scaling: ScalingType) -> np.ndarray:
    """Apply intensity scaling to per-segment intensities.

    Args:
        intensities: Array of shape (n_frames, n_segments).
        scaling: Scaling mode ('raw', 'norm', 'rel').

    Returns:
        Scaled intensity array.
    """
    if scaling == "raw":
        return intensities

    x = intensities.astype(float, copy=False)

    if scaling == "norm":
        # Per-frame min-max scaling to 0..100.
        # Frame axis is 0, segment axis is 1.
        mins = np.min(x, axis=1)
        maxs = np.max(x, axis=1)
        denom = (maxs - mins)

        out = np.zeros_like(x, dtype=float)
        # (x - min) / (max-min) * 100, with safe handling when denom==0
        np.subtract(x, mins[:, None], out=out)
        np.divide(out, denom[:, None], out=out, where=denom[:, None] != 0)
        out *= 100.0
        return out

    if scaling == "rel":
        # Per-frame relative intensity: segment_intensity / total_knee_intensity(frame)
        totals = np.sum(x, axis=1)
        out = np.zeros_like(x, dtype=float)
        np.divide(x, totals[:, None], out=out, where=totals[:, None] != 0)
        return out

    raise ValueError(f"Unknown scaling mode: {scaling!r}")


# =============================================================================
# CYCLE EXTRACTION AND AVERAGING STUBS
# =============================================================================

@dataclass
class CycleData:
    """Extracted cycle intensity data.

    Attributes:
        flex_mats: List of flexion intensity matrices (n_segments, n_frames_flex_i).
        ext_mats: List of extension intensity matrices (n_segments, n_frames_ext_i).
    """
    flex_mats: List[np.ndarray]
    ext_mats: List[np.ndarray]


def extract_cycles(
    intensities: np.ndarray,
    flexion_cycles: pd.DataFrame,
    extension_cycles: pd.DataFrame,
    cycle_indices: Optional[List[int]],
) -> CycleData:
    """Extract intensity matrices for selected cycles.

    Args:
        intensities: Full intensity matrix (n_frames, n_segments).
        flexion_cycles: DataFrame with flexion cycle frame ranges.
        extension_cycles: DataFrame with extension cycle frame ranges.
        cycle_indices: 1-based cycle indices to extract, or None for all.

    Returns:
        CycleData with extracted matrices.
    """
    n_frames, n_segments = intensities.shape

    def _extract_from(df_cycles: pd.DataFrame, idxs: List[int]) -> List[np.ndarray]:
        mats: List[np.ndarray] = []
        for cyc in idxs:
            row = df_cycles.iloc[cyc - 1]
            start = int(row["start"])  # 1-based inclusive
            end = int(row["end"])      # 1-based inclusive

            # Convert to python slice [start-1, end)
            s0 = start - 1
            e0 = end
            if s0 < 0 or e0 > n_frames:
                raise ValueError(
                    f"Cycle frame range out of bounds: cycle={cyc}, start={start}, end={end}, "
                    f"n_frames={n_frames}"
                )

            seg_by_frame = intensities[s0:e0, :]           # (L, n_segments)
            mats.append(seg_by_frame.T.copy())             # (n_segments, L)
        return mats

    flex_idxs = cycle_indices if cycle_indices is not None else list(range(1, len(flexion_cycles) + 1))
    ext_idxs = cycle_indices if cycle_indices is not None else list(range(1, len(extension_cycles) + 1))

    return CycleData(
        flex_mats=_extract_from(flexion_cycles, flex_idxs) if len(flexion_cycles) else [],
        ext_mats=_extract_from(extension_cycles, ext_idxs) if len(extension_cycles) else [],
    )


def average_cycles(cycle_data: CycleData) -> Tuple[np.ndarray, np.ndarray]:
    """Average cycle intensity matrices with resampling.

    Each cycle is resampled to the longest cycle length before averaging.

    Args:
        cycle_data: Extracted cycle matrices.

    Returns:
        Tuple of (avg_flex, avg_ext) matrices.
    """
    def _resample_and_average(mats: List[np.ndarray]) -> np.ndarray:
        if not mats:
            # Caller may be plotting a single phase; represent missing phase as empty.
            return np.zeros((0, 0), dtype=float)

        n_segments = int(mats[0].shape[0])
        if any(m.shape[0] != n_segments for m in mats):
            raise ValueError("Inconsistent n_segments across cycles")

        lengths = [int(m.shape[1]) for m in mats]
        l_max = max(lengths)
        if l_max <= 0:
            raise ValueError("Encountered cycle with non-positive length")

        x_new = np.linspace(0.0, 1.0, l_max)
        resampled = np.zeros((len(mats), n_segments, l_max), dtype=float)

        for i, m in enumerate(mats):
            l_i = int(m.shape[1])
            if l_i == l_max:
                resampled[i] = m
                continue
            if l_i < 2:
                # Degenerate: repeat the single value.
                resampled[i] = np.repeat(m, repeats=l_max, axis=1)
                continue
            x_old = np.linspace(0.0, 1.0, l_i)
            for s in range(n_segments):
                resampled[i, s, :] = np.interp(x_new, x_old, m[s, :])

        return np.mean(resampled, axis=0)

    avg_flex = _resample_and_average(cycle_data.flex_mats)
    avg_ext = _resample_and_average(cycle_data.ext_mats)
    return avg_flex, avg_ext


# =============================================================================
# METRIC COMPUTATION STUBS
# =============================================================================

def compute_com(avg_flex: np.ndarray, avg_ext: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute center-of-mass from averaged intensity matrices.

    Args:
        avg_flex: Averaged flexion intensity (n_segments, n_frames_flex).
        avg_ext: Averaged extension intensity (n_segments, n_frames_ext).

    Returns:
        Tuple of (com_flex, com_ext) arrays.
    """
    # TODO: Implement COM computation
    raise NotImplementedError("compute_com not yet implemented")


def compute_total(avg_flex: np.ndarray, avg_ext: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total intensity from averaged intensity matrices.

    Args:
        avg_flex: Averaged flexion intensity (n_segments, n_frames_flex).
        avg_ext: Averaged extension intensity (n_segments, n_frames_ext).

    Returns:
        Tuple of (total_flex, total_ext) arrays.
    """
    # TODO: Implement total intensity computation
    raise NotImplementedError("compute_total not yet implemented")


def compute_flux(
    avg_flex: np.ndarray,
    avg_ext: np.ndarray,
    anatomical_regions: pd.DataFrame,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Compute boundary flux from averaged intensity matrices.

    Args:
        avg_flex: Averaged flexion intensity (n_segments, n_frames_flex).
        avg_ext: Averaged extension intensity (n_segments, n_frames_ext).
        anatomical_regions: DataFrame with region boundaries.

    Returns:
        Tuple of ((flux_jc_ot_flex, flux_jc_ot_ext), (flux_ot_sb_flex, flux_ot_sb_ext)).
    """
    # TODO: Implement flux computation (reuse logic from dmm_analysis.py)
    raise NotImplementedError("compute_flux not yet implemented")


# =============================================================================
# PLOTTING STUBS
# =============================================================================

def plot_metric_angle_domain(
    video_data: List[Tuple[VideoSpec, np.ndarray, np.ndarray]],
    metric: MetricType,
    phase: PhaseType,
    n_interp_samples: int,
    out_path: Optional[Path],
    show: bool,
) -> None:
    """Plot metric curves in angle domain.

    Args:
        video_data: List of (VideoSpec, metric_flex, metric_ext) tuples.
        metric: Metric type being plotted.
        phase: Phase to plot.
        n_interp_samples: Number of interpolation samples per phase.
        out_path: Path to save PDF, or None.
        show: Whether to display interactively.
    """
    # TODO: Implement angle-domain plotting
    raise NotImplementedError("plot_metric_angle_domain not yet implemented")


def plot_metric_frame_domain(
    video_data: List[Tuple[VideoSpec, np.ndarray, np.ndarray]],
    metric: MetricType,
    phase: PhaseType,
    out_path: Optional[Path],
    show: bool,
) -> None:
    """Plot metric curves in frame domain.

    Args:
        video_data: List of (VideoSpec, metric_flex, metric_ext) tuples.
        metric: Metric type being plotted.
        phase: Phase to plot.
        out_path: Path to save PDF, or None.
        show: Whether to display interactively.
    """
    # TODO: Implement frame-domain plotting
    raise NotImplementedError("plot_metric_frame_domain not yet implemented")


# =============================================================================
# OUTPUT NAMING
# =============================================================================

def build_output_filename(
    metric: MetricType,
    phase: PhaseType,
    x_domain: XDomainType,
    scaling: ScalingType,
    source: SourceType,
    video_specs: List[VideoSpec],
) -> str:
    """Build output filename from parameters.

    Args:
        metric: Metric type.
        phase: Phase selection.
        x_domain: X-axis domain.
        scaling: Scaling mode.
        source: Data source.
        video_specs: Video specifications included in the plot.

    Returns:
        Filename stem (without extension).
    """

    def _video_token(spec: VideoSpec) -> str:
        """Readable identifier for a video in a multi-video plot.

        Prefer a canonical `<video_id:04d>N<n_segments>` token when possible.
        """
        if spec.video_id is not None and spec.n_segments is not None:
            return f"{int(spec.video_id):04d}N{int(spec.n_segments)}"
        # Fallback: strip extension.
        return re.sub(r"\.xlsx$", "", spec.base, flags=re.IGNORECASE)

    # Fully explicit list of videos.
    tokens = [_video_token(s) for s in video_specs]
    videos_part = "videos-" + "_".join(tokens)

    # Short stable tag to disambiguate runs that have the same set of videos but
    # differ in cycles, labels, order, or other spec-level settings.
    def _canonical_spec_string(spec: VideoSpec) -> str:
        cycles = "all" if spec.cycles is None else ",".join(str(i) for i in spec.cycles)
        label = "" if spec.label is None else spec.label
        return f"base={spec.base}|cycles={cycles}|label={label}"

    payload = "||".join(_canonical_spec_string(s) for s in video_specs)
    tag6 = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:6]

    # Partition ordering: averaged -> metric -> (plot parameters) -> (dataset partition)
    return (
        f"averaged_{metric}_{x_domain}_{phase}_scaling-{scaling}_source-{source}_"
        f"{videos_part}_tag{tag6}"
    )


# =============================================================================
# LIST MODE
# =============================================================================

def list_available_workbooks() -> None:
    """Print available workbook basenames under data/intensities_total/."""
    if not INTENSITIES_DIR.is_dir():
        print(f"Directory not found: {INTENSITIES_DIR}")
        return

    basenames = []
    for p in INTENSITIES_DIR.iterdir():
        if p.name.startswith(".~lock."):
            continue
        if p.suffix.lower() == ".xlsx":
            basenames.append(p.name)

    if not basenames:
        print(f"No Excel workbooks found in {INTENSITIES_DIR}")
        return

    print(f"Available workbooks in {INTENSITIES_DIR}:")
    for name in sorted(basenames):
        print(f"  {name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle --list mode
    if args.list_mode:
        list_available_workbooks()
        return 0

    # Resolve all VIDEO_SPEC paths
    for spec in args.video_specs:
        try:
            spec.resolved_path = resolve_video_spec_path(spec)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # One output per invocation (combined multi-video plot)
    out_stem = build_output_filename(
        metric=args.metric,
        phase=args.phase,
        x_domain=args.x_domain,
        scaling=args.scaling,
        source=args.source,
        video_specs=args.video_specs,
    )
    print(f"\nOutput: {out_stem}.pdf")

    # Process each item
    for spec in args.video_specs:
        print(f"\nProcessing {spec.base} (cycles={spec.cycles or 'all'}, label={spec.label or 'default'})")
        print(f"    Resolved: {spec.resolved_path}")

        # TODO: Implement full pipeline:
        # 1. Load workbooks
        # 2. Validate cycles
        # 3. Apply scaling
        # 4. Extract and average cycles
        # 5. Compute metric
        # 6. Plot

        # 1. Load workbooks
        workbook = load_workbook(spec, args.source)
        
        # 2. Validate cycles
        validate_cycles(spec, workbook, args.phase)

        # 3. Apply scaling
        scaled = apply_intensity_scaling(workbook.intensities, args.scaling)

        # 4. Extract and average cycles
        cycle_data = extract_cycles(
            intensities=scaled,
            flexion_cycles=workbook.flexion_cycles,
            extension_cycles=workbook.extension_cycles,
            cycle_indices=spec.cycles,
        )
        avg_flex, avg_ext = average_cycles(cycle_data)

        # Debug prints for pipeline progress (until metrics/plotting are implemented)
        if args.phase in ("flexion", "both"):
            print(f"    Averaged flexion shape: {avg_flex.shape}")
        if args.phase in ("extension", "both"):
            print(f"    Averaged extension shape: {avg_ext.shape}")

    print("\nPipeline not yet implemented. Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
