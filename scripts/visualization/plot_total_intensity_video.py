"""Plot total intensity vs frame for a single video workbook.

Loads per-segment intensity data from Excel workbooks in `data/intensities_total/` and
plots the summed (total) intensity per frame.

Interface mirrors the VIDEO_SPEC and --list behavior used by
[`scripts/visualization/plot_averaged_cycle_metrics.py`](scripts/visualization/plot_averaged_cycle_metrics.py:420).

Usage examples:
  # List available workbooks
  python scripts/visualization/plot_total_intensity_video.py --list

  # Plot one video (all cycles)
  python scripts/visualization/plot_total_intensity_video.py 1342N64

  # Plot with cycle annotations/subplots for cycles 1-3
  python scripts/visualization/plot_total_intensity_video.py 1342N64:cycles=1,2,3 --subplot-dims 2,2

Notes:
  - Excel temp/lock files like '.~lock.*' are ignored.
  - Cycle indices are 1-based in the CLI and mapped to workbook row indices.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import mplcursors

IntensityScaling = str  # Literal-like: {"raw","norm","rel"}


def apply_intensity_scaling(intensities: np.ndarray, scaling: IntensityScaling) -> np.ndarray:
    """Apply an intensity scaling mode to per-segment intensities.

    Ported from [`apply_intensity_scaling()`](scripts/visualization/dmm_analysis.py:281),
    adapted to this script's array layout.

    Parameters
    ----------
    intensities:
        Shape (n_frames, n_segments).
    scaling: {"raw", "norm", "rel"}
        - raw: no scaling
        - norm: per-frame min-max normalization (0..100)
        - rel: per-frame relative intensities: segment_intensity / total_knee_intensity
    """
    scaling = str(scaling).lower().strip()
    if scaling == "raw":
        return intensities

    x = intensities.astype(float, copy=False)

    if scaling == "norm":
        out = x.copy()
        mins = out.min(axis=1)
        maxs = out.max(axis=1)
        den = maxs - mins
        valid = den > 0
        out[valid] = 100.0 * (out[valid] - mins[valid, None]) / den[valid, None]
        out[~valid] = 0.0
        return out

    if scaling == "rel":
        totals = x.sum(axis=1)  # (n_frames,)
        out = np.zeros_like(x, dtype=float)
        np.divide(x, totals[:, None], out=out, where=totals[:, None] != 0)
        return out

    raise ValueError(f"Unexpected scaling mode: {scaling!r}")


# =============================================================================
# CONSTANTS
# =============================================================================

INTENSITIES_DIR = Path("data") / "intensities_total"
DEFAULT_OUT_DIR = Path("figures")

SHEET_SEGMENT_INTENSITIES = "Segment Intensities"
SHEET_SEGMENT_INTENSITIES_BGSUB = "Segment Intensities (bgsub)"
SHEET_FLEXION_FRAMES = "Flexion Frames"
SHEET_EXTENSION_FRAMES = "Extension Frames"


# =============================================================================
# VIDEO SPEC PARSING (ported/simplified from plot_averaged_cycle_metrics.py)
# =============================================================================


@dataclass
class VideoSpec:
    """Video workbook identifier.

    Attributes:
        base: Basename token or explicit filename.
        video_id: Parsed integer id if format like 1342N64.
        n_segments: Parsed segment count if format like 1342N64.
        cycles: Optional list of 1-based cycle indices to annotate/plot.
        resolved_path: Resolved xlsx path (filled after parsing).
    """

    base: str
    video_id: Optional[int] = None
    n_segments: Optional[int] = None
    # cycles semantics:
    #   - None => cycles KVARG not provided (do not annotate cycles)
    #   - "all" => annotate all cycles found in workbook
    #   - List[int] => annotate specified 1-based cycles
    cycles: Optional[object] = None  # None | "all" | List[int]
    resolved_path: Optional[Path] = None


def _parse_cycles_value(v: str) -> Optional[List[int]]:
    v = v.strip().lower()
    if v in ("",):
        return None
    if v == "all":
        # handled explicitly in parse_video_spec so we can distinguish from absent cycles
        return None
    out: List[int] = []
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as e:
            raise ValueError(f"Invalid cycles token: {tok!r}") from e
    if not out:
        return None
    if any(c <= 0 for c in out):
        raise ValueError(f"Cycles must be 1-based positive integers, got {out}")
    return out


def parse_video_spec(spec_str: str) -> VideoSpec:
    """Parse VIDEO_SPEC of the form:

      <base>[:cycles=1,2,3]

    where base can be:
      - '1342N64' (video id + segment count)
      - '1342N64intensities.xlsx' (explicit workbook filename)
    """
    if not spec_str or not spec_str.strip():
        raise ValueError("Empty VIDEO_SPEC")

    parts = [p.strip() for p in spec_str.split(":") if p.strip()]
    base = parts[0]
    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"Invalid KVARG {p!r}; expected key=value")
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    m = re.fullmatch(r"(?i)(\d+)N(\d+)", base)
    video_id = int(m.group(1)) if m else None
    n_segments = int(m.group(2)) if m else None

    cycles: Optional[object] = None
    if "cycles" in kv:
        v = kv["cycles"].strip().lower()
        if v == "all":
            cycles = "all"
        else:
            cycles = _parse_cycles_value(kv["cycles"])

    return VideoSpec(base=base, video_id=video_id, n_segments=n_segments, cycles=cycles)


def resolve_video_spec_path(spec: VideoSpec) -> Path:
    """Resolve the workbook path inside INTENSITIES_DIR."""
    # If user provided an explicit filename
    if spec.base.lower().endswith((".xlsx", ".xls")):
        path = INTENSITIES_DIR / spec.base
        if not path.is_file():
            raise FileNotFoundError(
                f"Workbook not found: {path}. Use --list to see available workbooks."
            )
        return path

    # If user provided an id+N token like 1342N64, assume <token>intensities.xlsx
    m = re.fullmatch(r"(?i)(\d+)N(\d+)", spec.base)
    if m:
        candidate = f"{m.group(1)}N{m.group(2)}intensities.xlsx"
        path = INTENSITIES_DIR / candidate
        if not path.is_file():
            # Fall back to any matching workbook that starts with token
            token = f"{m.group(1)}N{m.group(2)}"
            matches = sorted(
                p
                for p in INTENSITIES_DIR.glob(f"{token}*.xlsx")
                if not p.name.startswith(".~lock.")
            )
            if matches:
                return matches[0]
            raise FileNotFoundError(
                f"Workbook not found for token {spec.base!r}. Expected {candidate!r} under {INTENSITIES_DIR}. "
                "Use --list to see available workbooks."
            )
        return path

    # Otherwise interpret as a stem and match <stem>.xlsx
    path = INTENSITIES_DIR / f"{spec.base}.xlsx"
    if path.is_file():
        return path

    raise FileNotFoundError(
        f"Could not resolve VIDEO_SPEC base {spec.base!r} to an .xlsx file under {INTENSITIES_DIR}. "
        "Use --list to see available workbooks."
    )


# =============================================================================
# EXCEL LOADING
# =============================================================================


@dataclass
class WorkbookData:
    path: Path
    intensities: np.ndarray  # (n_frames, n_segments)
    flexion_cycles: pd.DataFrame  # columns: start,end (int)
    extension_cycles: pd.DataFrame  # columns: start,end (int)


def _clean_cycle_sheet(df_raw: pd.DataFrame, sheet: str, path: Path) -> pd.DataFrame:
    if df_raw.empty:
        raise ValueError(f"Sheet {sheet!r} in workbook {path} is empty")
    df = df_raw.copy().dropna(how="all")
    if df.empty:
        raise ValueError(f"Sheet {sheet!r} in workbook {path} has no data")

    # If the first row looks like a header, drop it.
    first_row = df.iloc[0, :].astype(str).str.lower()
    if ("start" in first_row.values) and ("end" in first_row.values):
        df = df.iloc[1:, :]

    # Keep first 3 columns max: [cycle?], start, end
    df = df.iloc[:, 0:3]
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


def load_workbook(path: Path, source: str) -> WorkbookData:
    intensity_sheet = (
        SHEET_SEGMENT_INTENSITIES if source == "raw" else SHEET_SEGMENT_INTENSITIES_BGSUB
    )
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        raise ValueError(f"Failed to open workbook {path}: {e}") from e

    required = [intensity_sheet, SHEET_FLEXION_FRAMES, SHEET_EXTENSION_FRAMES]
    missing = [s for s in required if s not in xls.sheet_names]
    if missing:
        raise FileNotFoundError(
            f"Workbook {path} is missing required sheets: {missing}. Available: {xls.sheet_names}"
        )

    df_int = pd.read_excel(xls, sheet_name=intensity_sheet, header=0, index_col=0)
    df_int = df_int.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    intensities = df_int.to_numpy(dtype=float)

    df_flex_raw = pd.read_excel(xls, sheet_name=SHEET_FLEXION_FRAMES, header=None)
    df_ext_raw = pd.read_excel(xls, sheet_name=SHEET_EXTENSION_FRAMES, header=None)
    flex = _clean_cycle_sheet(df_flex_raw, SHEET_FLEXION_FRAMES, path)
    ext = _clean_cycle_sheet(df_ext_raw, SHEET_EXTENSION_FRAMES, path)

    return WorkbookData(path=path, intensities=intensities, flexion_cycles=flex, extension_cycles=ext)


# =============================================================================
# LIST MODE
# =============================================================================


def list_available_workbooks() -> None:
    if not INTENSITIES_DIR.is_dir():
        print(f"Directory not found: {INTENSITIES_DIR}")
        return
    names = []
    for p in INTENSITIES_DIR.iterdir():
        if p.name.startswith(".~lock."):
            continue
        if p.suffix.lower() == ".xlsx":
            names.append(p.name)
    if not names:
        print(f"No Excel workbooks found in {INTENSITIES_DIR}")
        return
    print(f"Available workbooks in {INTENSITIES_DIR}:")
    for n in sorted(names):
        print(f"  {n}")


# =============================================================================
# PLOTTING
# =============================================================================


def _parse_subplot_dims(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[,x]\s*(\d+)\s*", s)
    if not m:
        raise ValueError("--subplot-dims must be like '2,3' or '2x3'")
    r, c = int(m.group(1)), int(m.group(2))
    if r <= 0 or c <= 0:
        raise ValueError("--subplot-dims must be positive")
    return r, c


def _cycle_span_frames(workbook: WorkbookData, cycle_idx_1based: int) -> Tuple[int, int]:
    i = cycle_idx_1based - 1
    if i < 0 or i >= len(workbook.flexion_cycles) or i >= len(workbook.extension_cycles):
        raise IndexError(
            f"Cycle {cycle_idx_1based} out of range. Flex cycles={len(workbook.flexion_cycles)}, "
            f"ext cycles={len(workbook.extension_cycles)}"
        )
    s = int(workbook.flexion_cycles.loc[i, "start"])
    e = int(workbook.extension_cycles.loc[i, "end"])
    return s, e


def _split_three_parts_indexwise(
    data: np.ndarray,
    region_ranges: Dict[str, Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Split segmentwise data into anatomical regions using index ranges.

    This mirrors the pattern in [`scripts/visualization/dmm_analysis.py`](scripts/visualization/dmm_analysis.py:1345)
    where cycle data is split into regions indexwise before computing per-region metrics.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (n_frames, n_segments), got shape={data.shape}")

    n_segments = int(data.shape[1])
    out: Dict[str, np.ndarray] = {}
    for region_name, (s, e) in region_ranges.items():
        if s < 0 or e < 0 or s > e or e > n_segments:
            raise ValueError(
                f"Invalid region range for {region_name!r}: ({s},{e}) with n_segments={n_segments}"
            )
        out[region_name] = data[:, s:e]
    return out


def _compute_region_intensities(
    intensities: np.ndarray,
    *,
    region_ranges: Dict[str, Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Per-frame intensity for each region (sum over segments in that region)."""
    regions = _split_three_parts_indexwise(intensities, region_ranges)
    return {k: np.sum(v, axis=1) for k, v in regions.items()}


def plot_total_intensity(
    *,
    spec: VideoSpec,
    workbook: WorkbookData,
    source: str,
    scaling: str,
    subplot_dims: Optional[Tuple[int, int]],
    out_path: Optional[Path],
    show: bool,
    regions: bool,
    no_total: bool,
) -> None:
    intens = apply_intensity_scaling(workbook.intensities, scaling)
    n_frames, n_segments = intens.shape
    total = np.sum(intens, axis=1)

    region_series: Optional[Dict[str, np.ndarray]] = None
    if regions:
        # If no explicit anatomical mapping is available in this script, split the segment axis
        # into three contiguous parts (indexwise), matching the intent of dmm_analysis.
        i1 = n_segments // 3
        i2 = (2 * n_segments) // 3
        region_ranges = {
            "SB": (i2, n_segments),
            "OT": (i1, i2),
            "JC": (0, i1),
        }
        region_series = _compute_region_intensities(intens, region_ranges=region_ranges)

    if no_total and region_series is None:
        raise ValueError(
            "--no-total omits the total intensity curve; with --regions disabled there would be nothing to plot. "
            "Enable --regions or omit --no-total."
        )

    cycles_label: str
    if spec.cycles is None:
        cycles_label = "(none)"
    elif spec.cycles == "all":
        cycles_label = "all"
    else:
        cycles_label = ",".join(str(c) for c in spec.cycles)  # type: ignore[arg-type]

    # Figure layout
    if spec.cycles is not None and subplot_dims is not None:
        r, c = subplot_dims
        fig = plt.figure(figsize=(14, 8))
        # Keep vertical whitespace tight: main plot + grid of per-cycle panels
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.4, 3.0], hspace=0.15)
        ax_main = fig.add_subplot(gs[0, 0])
        gs_sub = gs[1, 0].subgridspec(nrows=r, ncols=c, hspace=0.22, wspace=0.18)
        axes_sub = [fig.add_subplot(gs_sub[i, j]) for i in range(r) for j in range(c)]
    else:
        fig, ax_main = plt.subplots(figsize=(14, 5))
        axes_sub = []

    # Main plot
    x = np.arange(n_frames, dtype=int)
    if not no_total:
        ax_main.plot(x, total, lw=1.6, color="C0", label="total")
    if region_series is not None:
        for i, (name, y) in enumerate(region_series.items()):
            ax_main.plot(x, y, lw=1.1, alpha=0.9, color=f"C{i+1}", label=name)
        ax_main.legend(loc="upper right", fontsize=9, frameon=True)
    elif not no_total:
        # Keep legend behavior minimal: only show legend when the total curve is labeled.
        ax_main.legend(loc="upper right", fontsize=9, frameon=True)
    ax_main.set_xlabel("Frame index")
    ax_main.set_ylabel("Total intensity (sum over segments)")
    ax_main.grid(True, alpha=0.25)
    ax_main.set_xlim(0, n_frames - 1)

    title = "Total intensity vs frame"
    subtitle = (
        f"workbook={workbook.path.name} | source={source} | n_frames={n_frames} | "
        f"n_segments={n_segments} | cycles={cycles_label}"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=12)

    # Cycle annotations and optional subplots
    if spec.cycles is not None:
        # Expand cycles=all -> explicit list once we know workbook cycle count.
        if len(workbook.flexion_cycles) != len(workbook.extension_cycles):
            raise ValueError(
                "Flexion/extension cycle sheet row count mismatch: "
                f"flex={len(workbook.flexion_cycles)} ext={len(workbook.extension_cycles)}"
            )

        cycles_to_plot: List[int]
        if spec.cycles == "all":
            cycles_to_plot = list(range(1, len(workbook.flexion_cycles) + 1))
        else:
            cycles_to_plot = list(spec.cycles)  # type: ignore[arg-type]

        # Determine y-range for shading
        y0, y1 = ax_main.get_ylim()
        for idx, cyc in enumerate(cycles_to_plot):
            try:
                start, end = _cycle_span_frames(workbook, cyc)
            except Exception as e:
                raise ValueError(f"Failed to resolve cycle {cyc}: {e}") from e

            # Boundary lines
            ax_main.axvline(start, color="0.35", lw=0.8, alpha=0.8)
            ax_main.axvline(end, color="0.35", lw=0.8, alpha=0.8)

            # Shaded region
            ax_main.axvspan(start, end, color="0.85", alpha=0.35, zorder=0)

            # Cycle number label in the middle
            xm = 0.5 * (start + end)
            ax_main.text(
                xm,
                y0 + 0.92 * (y1 - y0),
                str(cyc),
                ha="center",
                va="center",
                fontsize=9,
                color="0.25",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", alpha=0.9),
            )

            # Subplot series
            if axes_sub:
                if idx >= len(axes_sub):
                    continue
                ax = axes_sub[idx]
                s = max(0, int(start))
                e = min(n_frames - 1, int(end))
                xs = np.arange(s, e + 1, dtype=int)
                ys = total[s : e + 1]
                if not no_total:
                    ax.plot(xs, ys, lw=1.2, color="C0")
                ax.set_title(f"Cycle {cyc}: frames {s}-{e}", fontsize=9)
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=8)
        # Hide unused subplots
        for j in range(len(cycles_to_plot), len(axes_sub)):
            axes_sub[j].axis("off")

    # Tighten outer margins while leaving room for the 2-line suptitle.
    # Avoid constrained_layout here since it tends to inflate spacing with nested grids.
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.85)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        mplcursors.cursor(multiple=True)
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


@dataclass
class CliArgs:
    video_spec: Optional[VideoSpec]
    list_mode: bool
    source: str
    scaling: str
    out_dir: Path
    save: bool
    show: bool
    subplot_dims: Optional[Tuple[int, int]]
    regions: bool
    no_total: bool


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot total intensity per frame from intensity Excel workbooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "VIDEO_SPEC examples:\n"
            "  1342N64\n"
            "  1342N64:cycles=1,2,3\n"
            "  1342N64intensities.xlsx:cycles=2\n"
        ),
    )
    p.add_argument(
        "video_spec",
        nargs="?",
        metavar="VIDEO_SPEC",
        help="Video spec (see examples). Required unless --list is used.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available workbook basenames under data/intensities_total/",
    )
    p.add_argument(
        "--source",
        choices=["raw", "bgsub"],
        default="raw",
        help="Data source sheet to use. (default: raw)",
    )
    p.add_argument(
        "--scaling",
        choices=["raw", "norm", "rel"],
        default="raw",
        help="Intensity scaling applied after loading the workbook. (default: raw)",
    )
    p.add_argument(
        "--subplot-dims",
        type=str,
        default=None,
        help="When cycles are specified, create per-cycle subplots in an RxC grid, e.g. '2,3' or '2x3'.",
    )
    p.add_argument(
        "--regions",
        action="store_true",
        help="Also plot per-region intensity traces in the main plot (3 indexwise regions).",
    )
    p.add_argument(
        "--no-total",
        action="store_true",
        help=(
            "Omit the total intensity curve. If used without --regions, nothing would be plotted and the script exits "
            "with an error."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures. (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=True,
        help="Save figure as PDF (default: enabled)",
    )
    p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save figure",
    )
    p.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display figure interactively (default: enabled)",
    )
    p.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display figure",
    )
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> CliArgs:
    p = build_arg_parser()
    a = p.parse_args(argv)

    if a.list:
        dims = _parse_subplot_dims(a.subplot_dims) if a.subplot_dims else None
        return CliArgs(
            video_spec=None,
            list_mode=True,
            source=a.source,
            scaling=a.scaling,
            out_dir=a.out_dir,
            save=a.save,
            show=a.show,
            subplot_dims=dims,
            regions=bool(a.regions),
            no_total=bool(a.no_total),
        )

    if not a.video_spec:
        p.error("VIDEO_SPEC is required (unless --list is used)")

    spec = parse_video_spec(a.video_spec)
    dims = _parse_subplot_dims(a.subplot_dims) if a.subplot_dims else None

    return CliArgs(
        video_spec=spec,
        list_mode=False,
        source=a.source,
        scaling=a.scaling,
        out_dir=a.out_dir,
        save=a.save,
        show=a.show,
        subplot_dims=dims,
        regions=bool(a.regions),
        no_total=bool(a.no_total),
    )


def build_output_filename(*, workbook_name: str, source: str) -> str:
    stem = Path(workbook_name).stem
    return f"total_intensity_frames_{stem}_source-{source}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_mode:
        list_available_workbooks()
        return 0

    assert args.video_spec is not None
    spec = args.video_spec
    try:
        spec.resolved_path = resolve_video_spec_path(spec)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        wb = load_workbook(spec.resolved_path, args.source)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_path: Optional[Path]
    if args.save:
        out_stem = build_output_filename(workbook_name=wb.path.name, source=args.source)
        out_path = args.out_dir / f"{out_stem}.pdf"
        print(f"\nOutput: {out_path}")
    else:
        out_path = None

    plot_total_intensity(
        spec=spec,
        workbook=wb,
        source=args.source,
        scaling=args.scaling,
        subplot_dims=args.subplot_dims,
        out_path=out_path,
        show=args.show,
        regions=args.regions,
        no_total=args.no_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

