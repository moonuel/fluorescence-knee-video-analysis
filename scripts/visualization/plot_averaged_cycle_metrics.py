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
    # TODO: Implement workbook loading
    raise NotImplementedError("load_workbook not yet implemented")


def validate_cycles(spec: VideoSpec, workbook: WorkbookData) -> None:
    """Validate that requested cycle indices are in range.

    Args:
        spec: VideoSpec with cycles to validate.
        workbook: Loaded workbook data.

    Raises:
        ValueError: If any cycle index is out of range.
    """
    # TODO: Implement cycle validation
    raise NotImplementedError("validate_cycles not yet implemented")


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
    # TODO: Implement scaling (reuse logic from dmm_analysis.py)
    raise NotImplementedError("apply_intensity_scaling not yet implemented")


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
    # TODO: Implement cycle extraction
    raise NotImplementedError("extract_cycles not yet implemented")


def average_cycles(cycle_data: CycleData) -> Tuple[np.ndarray, np.ndarray]:
    """Average cycle intensity matrices with resampling.

    Each cycle is resampled to the longest cycle length before averaging.

    Args:
        cycle_data: Extracted cycle matrices.

    Returns:
        Tuple of (avg_flex, avg_ext) matrices.
    """
    # TODO: Implement cycle averaging with resampling
    raise NotImplementedError("average_cycles not yet implemented")


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
    n_segments: int,
) -> str:
    """Build output filename from parameters.

    Args:
        metric: Metric type.
        phase: Phase selection.
        x_domain: X-axis domain.
        scaling: Scaling mode.
        source: Data source.
        n_segments: Segment count (for grouping).

    Returns:
        Filename stem (without extension).
    """
    return f"{metric}_{phase}_{x_domain}_scaling-{scaling}_source-{source}_N{n_segments}"


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

    # Group by segment count
    groups: Dict[int, List[VideoSpec]] = {}
    for spec in args.video_specs:
        if spec.n_segments is None:
            print(f"Error: Could not determine segment count for {spec.base!r}", file=sys.stderr)
            return 1
        groups.setdefault(spec.n_segments, []).append(spec)

    # Process each group
    for n_segs, specs in groups.items():
        print(f"\nProcessing N={n_segs} group ({len(specs)} video(s))...")

        # TODO: Implement full pipeline:
        # 1. Load workbooks
        # 2. Validate cycles
        # 3. Apply scaling
        # 4. Extract and average cycles
        # 5. Compute metric
        # 6. Plot

        for spec in specs:
            print(f"  - {spec.base} (cycles={spec.cycles or 'all'}, label={spec.label or 'default'})")
            print(f"    Resolved: {spec.resolved_path}")

        # Build output filename
        out_stem = build_output_filename(
            metric=args.metric,
            phase=args.phase,
            x_domain=args.x_domain,
            scaling=args.scaling,
            source=args.source,
            n_segments=n_segs,
        )
        print(f"  Output: {out_stem}.pdf")

    print("\nPipeline not yet implemented. Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
