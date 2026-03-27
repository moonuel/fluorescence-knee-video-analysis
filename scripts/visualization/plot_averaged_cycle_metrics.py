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

import matplotlib.pyplot as plt


def _parse_pair_of_floats(text: str, *, name: str) -> tuple[float, float]:
    """Parse a CLI argument in the form "A,B" into a (A, B) float tuple.

    Used for `--figsize` and `--ylim`.
    """
    raw = "" if text is None else str(text).strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2 or any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"{name} must be a single value in the form 'A,B' (e.g., '12,4'). Got: {text!r}"
        )
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{name} must contain numeric values in the form 'A,B' (e.g., '12,4'). Got: {text!r}"
        ) from e
    return (a, b)


def parse_figsize_arg(text: str) -> tuple[float, float]:
    return _parse_pair_of_floats(text, name="--figsize")


def parse_ylim_arg(text: str) -> tuple[float, float]:
    return _parse_pair_of_floats(text, name="--ylim")


# =============================================================================
# COM SUMMARY STATISTICS (ported from plot_com_cycles_from_heatmaps.py)
# =============================================================================

def compute_com_stats(com_series: pd.Series) -> Dict[str, float]:
    """Compute mean, sample SD, and excursion range of COM values.

    NaN values are ignored.

    Args:
        com_series: Series of COM values.

    Returns:
        Dict with keys: mean, sd, range.
    """
    values = com_series.values.astype(float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {"mean": np.nan, "sd": np.nan, "range": np.nan}

    mean_val = float(np.mean(values))
    sd_val = float(np.std(values, ddof=1))
    range_val = float(np.max(values) - np.min(values))

    return {"mean": mean_val, "sd": sd_val, "range": range_val}


def compute_oscillation_indices(com_series: pd.Series) -> Dict[str, float]:
    """Compute mean absolute frame-to-frame COM change for full/flex/ext.

    The input is expected to be indexed so that flexion samples have negative
    indices and extension samples have non-negative indices; the function sorts
    by index and splits accordingly.

    NaN values are ignored.

    Args:
        com_series: Series of COM values (sorted by index).

    Returns:
        Dict with keys: osc_full, osc_flex, osc_ext.
    """
    com_series = com_series.sort_index()

    flex_series = com_series[com_series.index < 0]
    ext_series = com_series[com_series.index >= 0]

    def _osc(values: np.ndarray) -> float:
        values = values.astype(float)
        values = values[~np.isnan(values)]
        if values.size < 2:
            return np.nan
        diffs = np.diff(values)
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            return np.nan
        return float(np.mean(np.abs(diffs)))

    osc_full = _osc(com_series.values)
    osc_flex = _osc(flex_series.values)
    osc_ext = _osc(ext_series.values)

    return {"osc_full": osc_full, "osc_flex": osc_flex, "osc_ext": osc_ext}


def build_signed_com_series(metric_flex: np.ndarray, metric_ext: np.ndarray, *, name: str) -> pd.Series:
    """Build a COM `pd.Series` with a signed integer index.

    Flexion samples are indexed as negative integers and extension samples as
    non-negative integers. This convention allows `compute_oscillation_indices()`
    to split flexion/extension by the index sign.
    """

    flex = np.asarray(metric_flex, dtype=float)
    ext = np.asarray(metric_ext, dtype=float)

    # index: flexion -> negative, extension -> non-negative
    flex_idx = np.arange(-flex.size, 0, dtype=int)
    ext_idx = np.arange(0, ext.size, dtype=int)

    return pd.Series(
        data=np.concatenate([flex, ext], axis=0),
        index=np.concatenate([flex_idx, ext_idx], axis=0),
        name=name,
        dtype=float,
    )


def append_com_stats_rows(
    com_stats_rows: list[dict[str, object]],
    *,
    spec,
    metric_flex: np.ndarray,
    metric_ext: np.ndarray,
) -> None:
    """Append COM statistics rows for raw and 50/50-rescaled curves."""

    cycles_label = "all" if spec.cycles is None else ",".join(str(i) for i in spec.cycles)
    video_label = (
        f"{int(spec.video_id):04d}N{int(spec.n_segments)}"
        if spec.video_id is not None and spec.n_segments is not None
        else spec.base
    )

    com_series = build_signed_com_series(metric_flex, metric_ext, name="com")
    stats = compute_com_stats(com_series)
    osc = compute_oscillation_indices(com_series)
    com_stats_rows.append({"video": video_label, "cycles": cycles_label, **stats, **osc})

    # Also compute stats on 50/50 equal-duration rescaled COM curves.
    flex_50, ext_50 = rescale_to_equal_duration_50_50(metric_flex, metric_ext)
    com_series_50 = build_signed_com_series(flex_50, ext_50, name="com_50_50")
    stats_50 = compute_com_stats(com_series_50)
    osc_50 = compute_oscillation_indices(com_series_50)
    com_stats_rows.append(
        {
            "video": f"{video_label} (50/50)",
            "cycles": cycles_label,
            **stats_50,
            **osc_50,
        }
    )


def rescale_to_equal_duration_50_50(
    flex: np.ndarray, ext: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample flexion and extension arrays to equal duration (50/50).

    Both phases are resampled to the same length L = max(len(flex), len(ext)).
    This mirrors the equal-duration (50/50) temporal rescaling used in
    [`scripts/analysis/generate_spatiotemporal_heatmaps.py:540`](scripts/analysis/generate_spatiotemporal_heatmaps.py:540),
    but applied to 1D COM curves.

    Args:
        flex: Flexion COM array (T_f,).
        ext: Extension COM array (T_e,).

    Returns:
        (flex_50, ext_50) each with shape (L,).
    """
    flex = np.asarray(flex, dtype=float)
    ext = np.asarray(ext, dtype=float)

    l_f, l_e = int(flex.size), int(ext.size)
    l = max(l_f, l_e)

    def _resample(x: np.ndarray, l_new: int) -> np.ndarray:
        if l_new <= 0:
            return np.asarray([], dtype=float)
        if x.size == 0:
            return np.full((l_new,), np.nan, dtype=float)
        if x.size == 1:
            return np.full((l_new,), float(x[0]), dtype=float)
        if x.size == l_new:
            return x.astype(float, copy=False)
        x_old = np.linspace(0.0, 1.0, x.size)
        x_new = np.linspace(0.0, 1.0, l_new)
        return np.interp(x_new, x_old, x.astype(float))

    return _resample(flex, l), _resample(ext, l)


def _format_float(x: float, ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{ndigits}f}"


def print_com_statistics_table(
    rows: List[Dict[str, Union[str, int, float]]],
    title: str = "COM summary statistics",
) -> None:
    """Print a simple fixed-width table of COM stats to the terminal."""
    if not rows:
        print(f"\n{title}: (no rows)")
        return

    columns = [
        "video",
        "cycles",
        "mean",
        "sd",
        "range",
        "osc_full",
        "osc_flex",
        "osc_ext",
    ]

    # Convert to strings first (for width computation)
    str_rows: List[Dict[str, str]] = []
    for r in rows:
        str_rows.append(
            {
                "video": str(r.get("video", "")),
                "cycles": str(r.get("cycles", "")),
                "mean": _format_float(float(r.get("mean", np.nan))),
                "sd": _format_float(float(r.get("sd", np.nan))),
                "range": _format_float(float(r.get("range", np.nan))),
                "osc_full": _format_float(float(r.get("osc_full", np.nan))),
                "osc_flex": _format_float(float(r.get("osc_flex", np.nan))),
                "osc_ext": _format_float(float(r.get("osc_ext", np.nan))),
            }
        )

    widths: Dict[str, int] = {}
    for c in columns:
        widths[c] = max(len(c), max(len(sr[c]) for sr in str_rows))

    def _row_line(values: Dict[str, str]) -> str:
        parts = [values[c].ljust(widths[c]) for c in columns]
        return "  ".join(parts)

    print(f"\n{title}:")
    print(_row_line({c: c for c in columns}))
    print(_row_line({c: "-" * widths[c] for c in columns}))
    for sr in str_rows:
        print(_row_line(sr))

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

Plot tuning example:
  python scripts/visualization/plot_averaged_cycle_metrics.py \
    1339N64:cycles=1,2,3 --metric com --phase both --scaling norm --source raw \
    --figsize 12,4 --title "My Plot" --ylim 0,100
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

    # Plot tuning
    parser.add_argument(
        "--figsize",
        type=parse_figsize_arg,
        default=None,
        help="Matplotlib figure size in inches, as W,H (e.g., 12,4). (default: unchanged)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title text. Use shell quoting for spaces. (default: unchanged)",
    )
    parser.add_argument(
        "--ylim",
        type=parse_ylim_arg,
        default=None,
        help="Y-axis limits as YMIN,YMAX (e.g., 0,100). (default: unchanged)",
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
    figsize: Optional[Tuple[float, float]]
    title: Optional[str]
    ylim: Optional[Tuple[float, float]]


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
            figsize=args.figsize,
            title=args.title,
            ylim=args.ylim,
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
        figsize=args.figsize,
        title=args.title,
        ylim=args.ylim,
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
    def _com(avg: np.ndarray) -> np.ndarray:
        # avg: (n_segments, n_frames)
        if avg.size == 0:
            return np.asarray([], dtype=float)
        n_segments, _ = avg.shape
        positions = np.arange(1, n_segments + 1, dtype=float)[:, None]  # (n_segments, 1)
        weighted = (positions * avg).sum(axis=0)
        totals = avg.sum(axis=0)
        com = np.divide(
            weighted,
            totals,
            out=np.full_like(weighted, np.nan, dtype=float),
            where=totals != 0,
        )
        return com

    return _com(avg_flex), _com(avg_ext)


def compute_total(avg_flex: np.ndarray, avg_ext: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total intensity from averaged intensity matrices.

    Args:
        avg_flex: Averaged flexion intensity (n_segments, n_frames_flex).
        avg_ext: Averaged extension intensity (n_segments, n_frames_ext).

    Returns:
        Tuple of (total_flex, total_ext) arrays.
    """
    def _total(avg: np.ndarray) -> np.ndarray:
        if avg.size == 0:
            return np.asarray([], dtype=float)
        return avg.sum(axis=0).astype(float, copy=False)

    return _total(avg_flex), _total(avg_ext)


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
        Tuple of ((flux_sb_ot_flex, flux_sb_ot_ext), (flux_ot_jc_flex, flux_ot_jc_ext)).
    """
    # Interpret anatomical_regions as 1-based inclusive segment index ranges.
    # Expected regions: JC, OT, SB (case-insensitive).
    df = anatomical_regions.copy()
    df["Region"] = df["Region"].astype(str)

    def _get_range(name: str) -> Tuple[int, int]:
        m = df[df["Region"].str.upper() == name.upper()]
        if m.empty:
            raise ValueError(
                f"Missing region {name!r} in Anatomical Regions sheet. "
                f"Found: {sorted(df['Region'].str.upper().unique().tolist())}"
            )
        row = m.iloc[0]
        s = int(row["Start"])
        e = int(row["End"])
        if s < 1 or e < 1 or s > e:
            raise ValueError(f"Invalid range for region {name!r}: Start={s}, End={e}")
        return s, e

    jc_s, jc_e = _get_range("JC")
    ot_s, ot_e = _get_range("OT")
    sb_s, sb_e = _get_range("SB")

    def _region_totals(avg: np.ndarray, s: int, e: int) -> np.ndarray:
        if avg.size == 0:
            return np.asarray([], dtype=float)
        n_segments = int(avg.shape[0])
        if e > n_segments:
            raise ValueError(
                f"Region range out of bounds for n_segments={n_segments}: Start={s}, End={e}"
            )
        return avg[s - 1 : e, :].sum(axis=0).astype(float, copy=False)

    def _boundary_fluxes(avg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (flux_sb_ot, flux_ot_jc) for a single phase.

        Convention (mirrors dmm_analysis.py):
        - SB->OT boundary: flux_sb_ot(t) = I_SB(t) - I_SB(t+1)
        - OT->JC boundary: flux_ot_jc(t) = I_JC(t+1) - I_JC(t)

        Both yield arrays of length (n_frames-1).
        """
        if avg.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        I_JC = _region_totals(avg, jc_s, jc_e)
        I_SB = _region_totals(avg, sb_s, sb_e)
        if I_JC.size < 2 or I_SB.size < 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        flux_sb_ot = I_SB[:-1] - I_SB[1:]
        flux_ot_jc = I_JC[1:] - I_JC[:-1]
        return flux_sb_ot, flux_ot_jc

    flux_sb_ot_flex, flux_ot_jc_flex = _boundary_fluxes(avg_flex)
    flux_sb_ot_ext, flux_ot_jc_ext = _boundary_fluxes(avg_ext)
    return (
        (flux_sb_ot_flex, flux_sb_ot_ext),
        (flux_ot_jc_flex, flux_ot_jc_ext),
    )


# =============================================================================
# PLOTTING STUBS
# =============================================================================

def plot_metric_angle_domain(
    video_data: List[Tuple[VideoSpec, np.ndarray, np.ndarray]],
    metric: MetricType,
    phase: PhaseType,
    scaling: ScalingType,
    n_interp_samples: int,
    out_path: Optional[Path],
    show: bool,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot metric curves in angle domain.

    Args:
        video_data: List of (VideoSpec, metric_flex, metric_ext) tuples.
        metric: Metric type being plotted.
        phase: Phase to plot.
        scaling: Scaling to apply.
        n_interp_samples: Number of interpolation samples per phase.
        out_path: Path to save PDF, or None.
        show: Whether to display interactively.
    """
    def _interp_1d(y: np.ndarray) -> np.ndarray:
        """Interpolate a 1D series to n_interp_samples over normalized x in [0, 1]."""
        if y.size == 0:
            return np.full((n_interp_samples,), np.nan, dtype=float)

        y = np.asarray(y, dtype=float)
        x_old = np.linspace(0.0, 1.0, num=y.shape[0])
        x_new = np.linspace(0.0, 1.0, num=n_interp_samples)

        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.full((n_interp_samples,), np.nan, dtype=float)

        y_new = np.interp(x_new, x_old[mask], y[mask])
        return y_new

    def _interp_2d_by_row(y2: np.ndarray) -> np.ndarray:
        """Interpolate a 2D (K x T) series to (K x n_interp_samples) over normalized x in [0, 1]."""
        if y2.size == 0:
            return np.full((y2.shape[0] if y2.ndim == 2 else 0, n_interp_samples), np.nan, dtype=float)

        y2 = np.asarray(y2, dtype=float)
        k, _t = y2.shape
        out = np.full((k, n_interp_samples), np.nan, dtype=float)
        for i in range(k):
            out_i = _interp_1d(y2[i, :])
            out[i, :] = out_i
        return out

    def _plot_series(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None:
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if m.any():
            ax.plot(x[m], y[m], label=label, **kwargs)

    # Plot in sample-index domain so we can label 30→135→30 explicitly.
    # If both phases are drawn, we concatenate flexion and extension on one axis:
    #   flexion indices:   0 .. n_interp_samples-1
    #   extension indices: n_interp_samples .. 2*n_interp_samples-1
    x_flex = np.arange(n_interp_samples, dtype=float)
    x_ext = np.arange(n_interp_samples, 2 * n_interp_samples, dtype=float)

    tick_angles = np.arange(30.0, 135.0 + 0.5 * 15.0, 15.0, dtype=float)  # 30..135 inclusive
    tick_labels = [f"{int(a)}°" for a in tick_angles]

    # Tick positions derived from the assumption that each phase spans n_interp_samples points
    # uniformly from 30..135 (flex) and 135..30 (ext).
    flex_tick_pos = (tick_angles - 30.0) / (135.0 - 30.0) * (n_interp_samples - 1)
    ext_tick_pos = n_interp_samples + (135.0 - tick_angles) / (135.0 - 30.0) * (n_interp_samples - 1)

    fig, ax = plt.subplots(figsize=figsize or (10, 5))
    ax.grid(True, alpha=0.3)

    # Metric-specific labeling
    if metric == "com":
        y_label = "COM (segment index)"
        title_metric = "Center of Mass"
    elif metric == "total":
        y_label = "Total intensity"
        title_metric = "Total Intensity"
    else:
        y_label = "Flux"
        title_metric = "Boundary Flux"

    if ylim is not None:
        ax.set_ylim(ylim)

    # Deterministic ordering: plot in input order
    # Use Matplotlib's default color cycle, but keep a single color per dataset
    # (i.e., flexion + extension share the same color).
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, (spec, metric_flex, metric_ext) in enumerate(video_data):
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        base_label = spec.label or spec.base

        "Quick patch: in COM mode, force specific video IDs to dashed lines."
        # Easiest implementation: match on digits in spec.base (e.g., '1339', '1342', '1358').
        dashed_ids = ("1339", "1342", "1358", "1357")
        use_dashed = metric == "com" and any(v in str(spec.base) for v in dashed_ids)
        ls = "--" if use_dashed else "-"

        if metric != "flux":
            if phase in ("flexion", "both"):
                y_f = _interp_1d(metric_flex)
                _plot_series(
                    ax,
                    x_flex,
                    y_f,
                    label=base_label,
                    linestyle=ls,
                    linewidth=2,
                    color=color,
                )

            if phase in ("extension", "both"):
                y_e = _interp_1d(metric_ext)
                _plot_series(
                    ax,
                    x_ext if phase == "both" else x_flex,
                    y_e,
                    label="_nolegend_",
                    linestyle=ls,
                    linewidth=2,
                    color=color,
                )

        else:
            # flux metric is stored as 2xT: [SB->OT; OT->JC]
            # Keep one color per dataset, but distinguish the two flux components by linestyle.
            # SB->OT: solid, OT->JC: dashed (same color).
            if phase in ("flexion", "both"):
                y2_f = _interp_2d_by_row(metric_flex)
                _plot_series(
                    ax,
                    x_flex,
                    y2_f[0, :],
                    label=f"{base_label} SB->OT",
                    linestyle="-",
                    linewidth=2,
                    color=color,
                )
                _plot_series(
                    ax,
                    x_flex,
                    y2_f[1, :],
                    label=f"{base_label} OT->JC",
                    linestyle=(0, (4, 2)),
                    linewidth=2,
                    color=color,
                )

            if phase in ("extension", "both"):
                y2_e = _interp_2d_by_row(metric_ext)
                x_plot = x_ext if phase == "both" else x_flex
                _plot_series(
                    ax,
                    x_plot,
                    y2_e[0, :],
                    label="_nolegend_",
                    linestyle="-",
                    linewidth=2,
                    color=color,
                )
                _plot_series(
                    ax,
                    x_plot,
                    y2_e[1, :],
                    label="_nolegend_",
                    linestyle=(0, (4, 2)),
                    linewidth=2,
                    color=color,
                )

    ax.set_xlabel("Knee Angle (°)")
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Averaged {title_metric} vs Knee Angle ({phase}, scaling={scaling})")

    if phase == "flexion":
        ax.set_xlim(0, n_interp_samples - 1)
        ax.set_xticks(flex_tick_pos)
        ax.set_xticklabels(tick_labels)
    elif phase == "extension":
        ax.set_xlim(0, n_interp_samples - 1)
        ax.set_xticks((135.0 - tick_angles) / (135.0 - 30.0) * (n_interp_samples - 1))
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xlim(0, 2 * n_interp_samples - 1)
        ax.set_xticks(np.concatenate([flex_tick_pos, ext_tick_pos]))
        ax.set_xticklabels(tick_labels + tick_labels)
        ax.axvline(n_interp_samples - 0.5, color="k", alpha=0.15, linewidth=1)

        # Visual cue for phase halves
        ax.text(
            0.25,
            0.98,
            "flex",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            alpha=0.7,
        )
        ax.text(
            0.75,
            0.98,
            "ext",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            alpha=0.7,
        )

    ax.legend(loc="best", frameon=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


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

    # Accumulate computed metric series for plotting (one entry per video)
    video_metric_data: List[Tuple[VideoSpec, np.ndarray, np.ndarray]] = []

    # Optional: COM statistics table rows (computed only in --metric com mode)
    com_stats_rows: List[Dict[str, Union[str, int, float]]] = []

    # Process each item
    for spec in args.video_specs:
        print(f"\nProcessing {spec.base} (cycles={spec.cycles or 'all'}, label={spec.label or 'default'})")
        print(f"    Resolved: {spec.resolved_path}")

        # 1. Load workbooks
        # 2. Validate cycles
        # 3. Apply scaling
        # 4. Extract and average cycles
        # 5. Compute metric
        # 6. Plot
        # 7. Save TODO

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

        # 5. Compute metric
        if args.metric == "com":
            metric_flex, metric_ext = compute_com(avg_flex, avg_ext)

            # -----------------------------------------------------------------
            # COM statistics (ported from plot_com_cycles_from_heatmaps.py)
            # Build a single series with a signed index so we can split flex/ext
            # by index sign in compute_oscillation_indices().
            # -----------------------------------------------------------------
            cycles_label = "all" if spec.cycles is None else ",".join(str(i) for i in spec.cycles)
            video_label = (
                f"{int(spec.video_id):04d}N{int(spec.n_segments)}"
                if spec.video_id is not None and spec.n_segments is not None
                else spec.base
            )
            append_com_stats_rows(
                com_stats_rows,
                spec=spec,
                metric_flex=metric_flex,
                metric_ext=metric_ext,
            )
        elif args.metric == "total":
            metric_flex, metric_ext = compute_total(avg_flex, avg_ext)
        elif args.metric == "flux":
            (flux_sb_ot_flex, flux_sb_ot_ext), (flux_ot_jc_flex, flux_ot_jc_ext) = compute_flux(
                avg_flex, avg_ext, workbook.anatomical_regions
            )

            def _stack_flux(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                # Represent flux as 2xT: [SB->OT; OT->JC] (matches dmm_analysis.py naming)
                if a.size == 0 and b.size == 0:
                    return np.zeros((2, 0), dtype=float)
                if a.shape != b.shape:
                    raise ValueError(f"Flux series length mismatch: {a.shape} vs {b.shape}")
                return np.vstack([a, b]).astype(float, copy=False)

            metric_flex, metric_ext = _stack_flux(flux_sb_ot_flex, flux_ot_jc_flex), _stack_flux(flux_sb_ot_ext, flux_ot_jc_ext)
        else:
            raise ValueError(f"Unexpected metric: {args.metric!r}")

        video_metric_data.append((spec, metric_flex, metric_ext))

        # Debug prints for pipeline progress (until plotting is implemented)
        print(f"    Metric='{args.metric}': flex shape={metric_flex.shape}, ext shape={metric_ext.shape}")

    # Print COM statistics table before plotting (terminal export for now)
    if args.metric == "com":
        print_com_statistics_table(com_stats_rows, title="COM statistics (averaged cycle metric)")

    # 6. Plot
    out_path: Optional[Path]
    if args.save:
        out_path = args.out_dir / f"{out_stem}.pdf"
    else:
        out_path = None

    if args.x_domain == "angle":
        plot_metric_angle_domain(
            video_data=video_metric_data,
            metric=args.metric,
            phase=args.phase,
            scaling=args.scaling,
            n_interp_samples=args.n_interp_samples,
            out_path=out_path,
            show=args.show,
            figsize=args.figsize,
            title=args.title,
            ylim=args.ylim,
        )
        return 0

    raise NotImplementedError("Only --x-domain angle is implemented for plotting")


if __name__ == "__main__":
    sys.exit(main())
