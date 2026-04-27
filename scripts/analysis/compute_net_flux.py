"""
Compute per-cycle net flux metrics from DMM knee video data.

Computes net flux (intensity change over a cycle phase) for SB and JC anatomical
regions across multiple knee videos. Results are exported to a multi-sheet Excel
workbook.

Usage:
    python scripts/analysis/compute_net_flux.py [--scaling rel|raw] [--source raw|bgsub]
                                               [--output OUTPUT.xlsx] INPUT1 [INPUT2 ...]

Input format:
    {video_id:04d}N64:cycles={'all'|1,2,3,...}

    video_id    -- 4-digit zero-padded integer
    N64         -- literal; fixes n_segments=64
    cycles=     -- 'all' for every available cycle, or comma-separated 1-based indices

Examples:
    python scripts/analysis/compute_net_flux.py 0001N64:cycles=all
    python scripts/analysis/compute_net_flux.py --scaling raw --source bgsub 0001N64:cycles=all
    python scripts/analysis/compute_net_flux.py --output results/net_flux.xlsx 0001N64:cycles=all

Output:
    A single Excel workbook with one sheet per input configuration.
    Each sheet contains per-cycle net flux values plus an aggregate 'avg' row.

Output columns:
    cycle_id, SB_flex, SB_ext, JC_flex, JC_ext, combined_flex, combined_ext

Sign conventions:
    SB_flex: I_SB(flex_start) - I_SB(flex_end) — positive = SB loses intensity
    SB_ext: I_SB(ext_start) - I_SB(ext_end)
    JC_flex: I_JC(flex_end) - I_JC(flex_start) — positive = JC gains intensity
    JC_ext: I_JC(ext_end) - I_JC(ext_start)
    combined_flex: (I_SB+I_JC)(flex_start) - (I_SB+I_JC)(flex_end)
    combined_ext: (I_SB+I_JC)(ext_start) - (I_SB+I_JC)(ext_end)
"""

from __future__ import annotations

import argparse
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.visualization.plot_total_intensity_video import (
    load_workbook,
    WorkbookData,
    apply_intensity_scaling,
    _split_three_parts_indexwise,
    resolve_video_spec_path,
    VideoSpec,
)

DEFAULT_OUTPUT = Path("net_flux_results.xlsx")


def sanitize_sheet_name(raw: str) -> str:
    """Sanitize a string for use as an Excel sheet name (31 char limit, no forbidden chars)."""
    for ch in r"[]:*?/\\":
        raw = raw.replace(ch, "_")
    if len(raw) > 31:
        raw = raw[:31]
    return raw


def parse_input_config(config_str: str) -> dict:
    """
    Parse a config string of the form '{video_id:04d}N64:cycles=spec'

    Returns:
        {
            'video_id': int,        # e.g., 1
            'video_id_str': str,    # e.g., '0001'
            'n_segments': 64,
            'cycles': 'all' | [1, 2, 3, ...],  # 1-based cycle indices
            'cycles_label': str,    # e.g., 'all' or '1,2,3'
            'sheet_name': str,      # sanitized, e.g., '0001N64_all'
        }
    """
    config_str = config_str.strip()

    pattern = re.compile(
        r"^(?P<video_id>\d{4})"
        r"N(?P<n_segments>\d+)"
        r":cycles="
        r"(?P<cycles>all|\d+(?:,\d+)*)$",
        re.IGNORECASE,
    )

    match = pattern.match(config_str)
    if not match:
        raise ValueError(
            f"Invalid config string: {config_str!r}. "
            f"Expected format: '{{video_id:04d}}N{{n_segments}}:cycles={{all|1,2,3,...}}'. "
            f"Example: '0001N64:cycles=all' or '0010N64:cycles=1,2'"
        )

    video_id_str = match.group("video_id")
    video_id = int(video_id_str)
    n_segments = int(match.group("n_segments"))
    cycles_raw = match.group("cycles")

    if cycles_raw.lower() == "all":
        cycles: str | List[int] = "all"
    else:
        cycles = [int(c) for c in cycles_raw.split(",")]

    cycles_label = cycles_raw.lower()
    sheet_name = sanitize_sheet_name(f"{video_id_str}N{n_segments}_{cycles_label}")

    return {
        "video_id": video_id,
        "video_id_str": video_id_str,
        "n_segments": n_segments,
        "cycles": cycles,
        "cycles_label": cycles_label,
        "sheet_name": sheet_name,
    }


def _resolve_workbook_path(video_id: int, n_segments: int) -> Path:
    """Resolve the workbook path for a given video_id and n_segments."""
    spec = VideoSpec(
        base=f"{video_id:04d}N{n_segments}",
        video_id=video_id,
        n_segments=n_segments,
    )
    return resolve_video_spec_path(spec)


def _load_region_ranges(path: Path) -> Dict[str, Tuple[int, int]]:
    """Load anatomical region ranges from the workbook's 'Anatomical Regions' sheet.

    Returns dict mapping region name to (start_0based, end_exclusive) tuples.
    """
    try:
        df_regions = pd.read_excel(path, sheet_name="Anatomical Regions")
    except Exception as e:
        raise ValueError(
            f"Failed to load 'Anatomical Regions' sheet from {path}: {e}"
        ) from e

    region_ranges: Dict[str, Tuple[int, int]] = {}
    for _, row in df_regions.iterrows():
        region_name = str(row["Region"]).strip()
        start_excel = int(row["Start"])
        end_excel = int(row["End"])
        start_0based = start_excel - 1
        end_exclusive = end_excel
        region_ranges[region_name] = (start_0based, end_exclusive)

    return region_ranges


def compute_intensity_and_net_flux(config: dict, scaling: str, source: str) -> pd.DataFrame:
    """
    Load workbook, compute per-cycle net flux, return DataFrame.

    Parameters
    ----------
    config : dict
        Output of parse_input_config().
    scaling : str
        'rel' (relative intensities) or 'raw' (raw pixel sums).
    source : str
        'raw' or 'bgsub' — which intensity sheet to load from the workbook.

    Returns
    -------
    pd.DataFrame
        Columns: ['cycle_id', 'SB_flex', 'SB_ext', 'JC_flex', 'JC_ext']
    """
    video_id = config["video_id"]
    n_segments = config["n_segments"]

    path = _resolve_workbook_path(video_id, n_segments)
    logging.info("Loading workbook: %s", path)

    wb = load_workbook(path, source=source)

    region_ranges = _load_region_ranges(path)
    if "SB" not in region_ranges or "JC" not in region_ranges:
        raise ValueError(
            f"Workbook must contain 'SB' and 'JC' regions. Found: {list(region_ranges.keys())}"
        )

    logging.debug("Region ranges (0-based, end exclusive): %s", region_ranges)

    scaled_intensities = apply_intensity_scaling(wb.intensities, scaling)

    total_cycles = len(wb.flexion_cycles)
    if len(wb.extension_cycles) != total_cycles:
        raise ValueError(
            f"Flexion/extension cycle count mismatch: flex={total_cycles}, ext={len(wb.extension_cycles)}"
        )

    if config["cycles"] == "all":
        cycle_indices: List[int] = list(range(1, total_cycles + 1))
    else:
        cycle_indices = config["cycles"]
        for c in cycle_indices:
            if c < 1 or c > total_cycles:
                raise ValueError(
                    f"Cycle index {c} out of range [1, {total_cycles}] for {path.name}"
                )

    logging.info("Computing net flux for %d cycles (source=%s, scaling=%s)", len(cycle_indices), source, scaling)

    rows = []
    for cycle_num in cycle_indices:
        idx = cycle_num - 1

        flex_start_excel = int(wb.flexion_cycles.loc[idx, "start"])
        flex_end_excel = int(wb.flexion_cycles.loc[idx, "end"])
        ext_start_excel = int(wb.extension_cycles.loc[idx, "start"])
        ext_end_excel = int(wb.extension_cycles.loc[idx, "end"])

        flex_start = flex_start_excel - 1
        flex_end = flex_end_excel - 1
        ext_start = ext_start_excel - 1
        ext_end = ext_end_excel - 1

        flex_data = scaled_intensities[flex_start : flex_end + 1, :]
        ext_data = scaled_intensities[ext_start : ext_end + 1, :]

        flex_regions = _split_three_parts_indexwise(flex_data, region_ranges)
        ext_regions = _split_three_parts_indexwise(ext_data, region_ranges)

        I_SB_flex = np.sum(flex_regions["SB"], axis=1)
        I_JC_flex = np.sum(flex_regions["JC"], axis=1)
        I_SB_ext = np.sum(ext_regions["SB"], axis=1)
        I_JC_ext = np.sum(ext_regions["JC"], axis=1)

        SB_flex = float(I_SB_flex[0] - I_SB_flex[-1])
        JC_flex = float(I_JC_flex[-1] - I_JC_flex[0])
        SB_ext = float(I_SB_ext[0] - I_SB_ext[-1])
        JC_ext = float(I_JC_ext[-1] - I_JC_ext[0])

        rows.append({
            "cycle_id": cycle_num,
            "SB_flex": SB_flex,
            "SB_ext": SB_ext,
            "JC_flex": JC_flex,
            "JC_ext": JC_ext,
        })

        logging.debug(
            "Cycle %d: SB_flex=%.6f, SB_ext=%.6f, JC_flex=%.6f, JC_ext=%.6f",
            cycle_num, SB_flex, SB_ext, JC_flex, JC_ext
        )

    return pd.DataFrame(rows)


def aggregate_fluxes(per_cycle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append an 'avg' summary row and 'combined_flex'/'combined_ext' columns.

    Returns the DataFrame with summary row appended.
    """
    df = per_cycle_df.copy()

    numeric_cols = ["SB_flex", "SB_ext", "JC_flex", "JC_ext"]

    df["combined_flex"] = df["SB_flex"] + df["JC_flex"]
    df["combined_ext"] = df["SB_ext"] + df["JC_ext"]

    avg_row: Dict[str, object] = {"cycle_id": "avg"}
    for col in numeric_cols + ["combined_flex", "combined_ext"]:
        avg_row[col] = df[col].mean()

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    return df


def write_excel_workbook(sheet_dfs: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """
    Write each DataFrame to its own sheet in a single Excel workbook.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheet_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logging.info("Saved Excel workbook to %s", output_path)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-cycle net flux from knee DMM videos and export to Excel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/analysis/compute_net_flux.py 0001N64:cycles=all\n"
            "  python scripts/analysis/compute_net_flux.py --scaling raw --source bgsub 0001N64:cycles=all\n"
            "  python scripts/analysis/compute_net_flux.py --output results/net_flux.xlsx 0001N64:cycles=all\n"
        ),
    )

    parser.add_argument(
        "--scaling",
        choices=["rel", "raw"],
        default="rel",
        help="Intensity scaling: rel (relative fractions, default) or raw (pixel sums)",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "bgsub"],
        default="raw",
        help="Data source sheet: raw (default) or bgsub (background-subtracted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Excel workbook path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input configs, e.g. 0001N64:cycles=all 0010N64:cycles=1,2,3",
    )

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sheet_dfs: Dict[str, pd.DataFrame] = {}

    for config_str in args.inputs:
        try:
            config = parse_input_config(config_str)
        except ValueError as e:
            logging.error("Skipping invalid input %r: %s", config_str, e)
            continue

        try:
            per_cycle_df = compute_intensity_and_net_flux(config, args.scaling, args.source)
            full_df = aggregate_fluxes(per_cycle_df)
            sheet_dfs[config["sheet_name"]] = full_df
            logging.info("Processed %s: %d cycles -> sheet %r", config_str, len(per_cycle_df), config["sheet_name"])
        except Exception as e:
            logging.error("Failed to process %r: %s", config_str, e)
            continue

    if not sheet_dfs:
        logging.error("No valid inputs produced data. Exiting.")
        return 1

    write_excel_workbook(sheet_dfs, args.output)
    print(f"Saved {len(sheet_dfs)} sheet(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())