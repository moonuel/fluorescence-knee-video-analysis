"""One-off flux smoothing test plot.

Goal
----
Compare *unsmoothed* (dotted) vs *smoothed* (solid) boundary flux time series
for a single knee video, using the same angle/cycle plotting machinery as
`scripts/visualization/dmm_analysis.py`.

Variants (run the script multiple times)
--------------------------------------
1) Low-pass Butterworth filters with cutoff in {0.1, 0.25, 0.5}
2) Rolling average with window in {2, 5, 10}

Example invocation
--------------------------------
python scripts/visualization/plot_flux_smoothing_variants.py --metric flux --scaling raw --no-preview normal 0308 64 --cycles 1 --variant butter:0.25
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

# Reuse existing helpers from dmm_analysis.py (no refactor of that file required)
from scripts.visualization.dmm_analysis import (  # noqa: E402
    RegionRange,
    compute_boundary_flux,
    compute_intensity_data,
    construct_filename,
    get_knee_meta,
    load_video_data,
    normalize_intensity_per_frame_2d,
    parse_cycles_arg,
    split_three_parts_indexwise,
    build_angle_axis_for_cycles,
    build_angle_ticks,
    interpolate_series_to_angle,
)


@dataclass(frozen=True)
class FluxSmoothingVariant:
    kind: str  # "butter" or "rolling"
    value: float | int

    @property
    def label(self) -> str:
        if self.kind == "butter":
            return f"butter_lowpass_fc={self.value}"
        if self.kind == "rolling":
            return f"rolling_mean_window={self.value}"
        return f"{self.kind}:{self.value}"


def _smooth_intensity_series(x: np.ndarray, variant: FluxSmoothingVariant) -> np.ndarray:
    """Smooth a 1D intensity time series."""
    x = np.asarray(x, dtype=float)
    if variant.kind == "butter":
        # NOTE: scipy.signal.butter Wn is normalized to Nyquist (0..1)
        b, a = scipy.signal.butter(N=1, Wn=float(variant.value), btype="low", analog=False)
        return scipy.signal.filtfilt(b, a, x)
    if variant.kind == "rolling":
        w = int(variant.value)
        return pd.Series(x).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=float)
    raise ValueError(f"Unknown smoothing variant: {variant}")


def compute_flux_unsmoothed_and_smoothed(
    I_SB: np.ndarray,
    I_JC: np.ndarray,
    variant: FluxSmoothingVariant,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute flux from raw vs smoothed intensities.

    Returns
    -------
    (flux_raw, flux_smoothed)
        Each matches the dict format from `compute_boundary_flux()`.
    """
    flux_raw = compute_boundary_flux(I_SB, I_JC)
    I_SB_s = _smooth_intensity_series(I_SB, variant)
    I_JC_s = _smooth_intensity_series(I_JC, variant)
    flux_s = compute_boundary_flux(I_SB_s, I_JC_s)
    return flux_raw, flux_s


def _interp_cycle_block(series: np.ndarray, meta, cycle_idx: int, phase: str, n_interp_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a (frame) series for one cycle into an angle-parametrized block."""
    cyc = meta.cycles[cycle_idx]
    if phase == "flexion":
        frames = series[cyc.flex.s : cyc.flex.e + 1]
        vals, ang = interpolate_series_to_angle(frames, n_interp_samples, 30, 135)
        return np.asarray(vals), np.asarray(ang)
    if phase == "extension":
        frames = series[cyc.ext.s : cyc.ext.e + 1]
        vals, ang = interpolate_series_to_angle(frames, n_interp_samples, 135, 30)
        return np.asarray(vals), np.asarray(ang)
    # both
    frames_f = series[cyc.flex.s : cyc.flex.e + 1]
    frames_e = series[cyc.ext.s : cyc.ext.e + 1]
    vals_f, ang_f = interpolate_series_to_angle(frames_f, n_interp_samples, 30, 135)
    vals_e, ang_e = interpolate_series_to_angle(frames_e, n_interp_samples, 135, 30)
    return np.concatenate([vals_f, vals_e]), np.concatenate([ang_f, ang_e])


def _concat_cycles(series: np.ndarray, meta, cycle_indices: List[int | None], phase: str, n_interp_samples: int) -> np.ndarray:
    """Concatenate interpolated cycle blocks; None cycles become NaN blocks."""
    blocks: List[np.ndarray] = []
    for cidx in cycle_indices:
        if cidx is None:
            block_len = n_interp_samples if phase in {"flexion", "extension"} else 2 * n_interp_samples
            blocks.append(np.full((block_len,), np.nan, dtype=float))
        else:
            vals, _ang = _interp_cycle_block(series, meta, cidx, phase, n_interp_samples)
            blocks.append(vals.astype(float))
    return np.concatenate(blocks) if blocks else np.array([], dtype=float)


def plot_flux_overlay(
    flux_raw: Dict[str, np.ndarray],
    flux_smoothed: Dict[str, np.ndarray],
    meta,
    cycle_indices: List[int | None],
    phase: str,
    n_interp_samples: int,
    title: str,
    save_path: str | None,
    preview: bool,
) -> None:
    """Plot SB->OT and OT->JC, dotted raw vs solid smoothed."""
    # Flux arrays are (n_frames-1,), so pad to align with frame indices if needed.
    def pad_flux(f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return np.concatenate([f, [np.nan]])  # match n_frames length for slicing into cycles

    raw_sb = pad_flux(flux_raw["SB->OT"])
    raw_jc = pad_flux(flux_raw["OT->JC"])
    sm_sb = pad_flux(flux_smoothed["SB->OT"])
    sm_jc = pad_flux(flux_smoothed["OT->JC"])

    y_raw_sb = _concat_cycles(raw_sb, meta, cycle_indices, phase, n_interp_samples)
    y_raw_jc = _concat_cycles(raw_jc, meta, cycle_indices, phase, n_interp_samples)
    y_sm_sb = _concat_cycles(sm_sb, meta, cycle_indices, phase, n_interp_samples)
    y_sm_jc = _concat_cycles(sm_jc, meta, cycle_indices, phase, n_interp_samples)

    x_positions_list, _cycle_angles, _cycle_lengths = build_angle_axis_for_cycles(
        cycle_indices=cycle_indices,
        meta=meta,
        phase=phase,
        n_interp_samples=n_interp_samples,
    )
    x = np.concatenate(x_positions_list) if x_positions_list else np.array([])

    xticks, xticklabels = build_angle_ticks(phase, n_interp_samples, x_positions_list)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y_raw_sb, linestyle=":", linewidth=1.5, label="SB->OT raw", color="tab:red")
    ax.plot(x, y_sm_sb, linestyle="-", linewidth=2.0, label="SB->OT smoothed", color="tab:red")
    ax.plot(x, y_raw_jc, linestyle=":", linewidth=1.5, label="OT->JC raw", color="tab:blue")
    ax.plot(x, y_sm_jc, linestyle="-", linewidth=2.0, label="OT->JC smoothed", color="tab:blue")
    ax.axhline(0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Flux")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if preview:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("video_id")
    ap.add_argument("n_segments", type=int)
    ap.add_argument("--metric", default="flux")
    ap.add_argument("--cycles", default=None)
    ap.add_argument("--phase", default="both", choices=["flexion", "extension", "both"])
    ap.add_argument("--n-interp-samples", default=60, type=int)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument(
        "--scaling",
        default="raw",
        choices=["raw", "norm", "rel"],
        help=(
            "Intensity scaling mode: "
            "raw = no normalization (equivalent to the old --no-normalize), "
            "norm = per-frame normalization (equivalent to --normalize), "
            "rel = relative (percentaged) regional intensities per frame, i.e. region_sum / total_knee_sum."
        ),
    )
    ap.add_argument(
        "--variant",
        required=True,
        help="One of: butter:0.1 butter:0.25 butter:0.5 rolling:2 rolling:5 rolling:10",
    )
    args = ap.parse_args()

    if args.metric != "flux":
        raise ValueError("This one-off script only supports --metric flux")

    kind, val = args.variant.split(":", 1)
    variant = FluxSmoothingVariant(kind=kind, value=float(val) if kind == "butter" else int(val))

    cycle_indices, cycles_str = parse_cycles_arg(args.cycles)

    meta = get_knee_meta(args.condition, args.video_id, args.n_segments)

    video, masks = load_video_data(args.condition, args.video_id, args.n_segments)
    total_sums, _total_nonzero, _segment_labels = compute_intensity_data(video, masks)

    if args.scaling == "raw":
        # Equivalent to the old --no-normalize flag.
        normalization = "raw"
    elif args.scaling == "norm":
        # Equivalent to "normalize" (the previous default when --no-normalize was absent).
        total_sums = normalize_intensity_per_frame_2d(total_sums)
        normalization = "norm"
    elif args.scaling == "rel":
        # Relative intensities (percentaged) are applied later, after splitting into regions.
        normalization = "rel"
    else:
        # Defensive (argparse choices should prevent this).
        raise ValueError(f"Unexpected --scaling value: {args.scaling!r}")

    # Split into regions, then take totals for SB and JC (same logic as dmm_analysis compute_region_metrics)
    # Region metadata is stored on meta.regions (with .s/.e in 1-based indices)
    region_ranges = [
        RegionRange("JC", meta.regions["JC"].s, meta.regions["JC"].e),
        RegionRange("OT", meta.regions["OT"].s, meta.regions["OT"].e),
        RegionRange("SB", meta.regions["SB"].s, meta.regions["SB"].e),
    ]
    region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

    I_SB = region_arrays["SB"].sum(axis=0)
    I_JC = region_arrays["JC"].sum(axis=0)

    if args.scaling == "rel":
        # MATLAB equivalent:
        #   SRI = sum(RI, 2)
        #   pSRI_SB = SRI_SB ./ SRI
        #   pSRI_JC = SRI_JC ./ SRI
        # Here, per-frame "RI" corresponds to per-segment intensities; SRI is total knee brightness per frame.
        I_total = total_sums.sum(axis=0).astype(float)
        I_SB = np.divide(I_SB.astype(float), I_total, out=np.zeros_like(I_total, dtype=float), where=I_total != 0)
        I_JC = np.divide(I_JC.astype(float), I_total, out=np.zeros_like(I_total, dtype=float), where=I_total != 0)

    flux_raw, flux_smoothed = compute_flux_unsmoothed_and_smoothed(I_SB, I_JC, variant)

    modifiers = ["flux_smoothing", variant.label, "angle_based", f"phase_{args.phase}"]
    fname = construct_filename(
        analysis_type="boundary_flux",
        meta=meta,
        normalization=normalization,
        cycles=cycles_str,
        modifiers=modifiers,
        extension="png",
    )
    save_path = str((meta.figures_dir / fname)) if hasattr(meta, "figures_dir") else str(fname)

    title = (
        f"Flux smoothing comparison ({variant.label}) | scaling={args.scaling} | "
        f"{args.condition}_{args.video_id} N{args.n_segments}"
    )
    plot_flux_overlay(
        flux_raw=flux_raw,
        flux_smoothed=flux_smoothed,
        meta=meta,
        cycle_indices=cycle_indices,
        phase=args.phase,
        n_interp_samples=args.n_interp_samples,
        title=title,
        save_path=save_path,
        preview=(not args.no_preview),
    )


if __name__ == "__main__":
    main()

