import numpy as np


def test_build_output_filename_includes_avg_cycle_mode_and_scaling() -> None:
    from scripts.visualization.plot_averaged_cycle_metrics import build_output_filename

    # Minimal spec object matching what build_output_filename reads
    Spec = type("Spec", (), {})
    s = Spec()
    s.base = "0308N64"
    s.video_id = 308
    s.n_segments = 64
    s.cycles = [1, 2, 3]
    s.label = None

    stem = build_output_filename(
        avg_cycle_mode="averaged_metric",
        metric="total",
        phase="both",
        x_domain="angle",
        scaling="rel",
        source="raw",
        video_specs=[s],
    )
    assert stem.startswith("averaged_metric_total_angle_both_scaling-rel_source-raw_videos-")
    assert "_tag" in stem



def test_rel_scaling_percent_axis_label_and_formatter() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.visualization.plot_averaged_cycle_metrics import plot_metric_angle_domain

    try:
        spec = type("Spec", (), {"label": "v1", "base": "v1"})()
        # rel mode should append (%) and multiply tick labels by 100
        fig = plot_metric_angle_domain(
            video_data=[(spec, np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.1, 0.0]))],
            metric="total",
            phase="flexion",
            scaling="rel",
            n_interp_samples=5,
            out_path=None,
            show=False,
        )
        ax = fig.axes[0]
        assert ax.get_ylabel().endswith("(%)")

        fmt = ax.yaxis.get_major_formatter()
        assert fmt(0.2, 0) == "20"
    finally:
        plt.close("all")


def test_non_rel_scaling_no_percent_axis_label_suffix() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.visualization.plot_averaged_cycle_metrics import plot_metric_angle_domain

    try:
        spec = type("Spec", (), {"label": "v1", "base": "v1"})()
        fig = plot_metric_angle_domain(
            video_data=[(spec, np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 0.0]))],
            metric="total",
            phase="flexion",
            scaling="raw",
            n_interp_samples=5,
            out_path=None,
            show=False,
        )
        ax = fig.axes[0]
        assert not ax.get_ylabel().endswith("(%)")
    finally:
        plt.close("all")


def test_cycles_annotation_text_matches_input() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.visualization.plot_averaged_cycle_metrics import plot_metric_angle_domain

    try:
        spec = type("Spec", (), {"label": "v1", "base": "v1"})()
        fig = plot_metric_angle_domain(
            video_data=[(spec, np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 0.0]))],
            metric="total",
            phase="flexion",
            scaling="raw",
            n_interp_samples=5,
            out_path=None,
            show=False,
            cycles_used=[1, 3, 7],
        )

        # Annotation is now inside the axes (above x-axis), not fig-level.
        texts = [t.get_text() for t in fig.axes[0].texts]
        cycle_texts = [t for t in texts if t.startswith("Cycles ")]
        assert len(cycle_texts) == 1
        assert cycle_texts[0] == "Cycles 1,3,7"
    finally:
        plt.close("all")


def test_resample_series_to_common_grid_1d_basic() -> None:
    # Local import to avoid importing matplotlib during test collection in some environments
    from scripts.visualization.plot_averaged_cycle_metrics import _resample_series_to_common_grid

    y = np.array([0.0, 1.0, 2.0], dtype=float)
    out = _resample_series_to_common_grid(y, 5)
    assert out.shape == (5,)
    assert np.isfinite(out).all()
    # Endpoints preserved under linear interpolation
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[-1], 2.0)


def test_resample_series_to_common_grid_2d_basic() -> None:
    from scripts.visualization.plot_averaged_cycle_metrics import _resample_series_to_common_grid

    y = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=float)
    out = _resample_series_to_common_grid(y, 4)
    assert out.shape == (2, 4)
    assert np.isfinite(out).all()


def test_resample_series_to_common_grid_nan_policy() -> None:
    from scripts.visualization.plot_averaged_cycle_metrics import _resample_series_to_common_grid

    y = np.array([np.nan, 1.0, np.nan, 3.0], dtype=float)
    out = _resample_series_to_common_grid(y, 6)
    assert out.shape == (6,)
    # Should be finite due to two finite anchors
    assert np.isfinite(out).any()


def test_compute_metric_averaged_metric_shapes_total() -> None:
    from scripts.visualization.plot_averaged_cycle_metrics import (
        CycleData,
        compute_metric_averaged_metric,
    )
    import pandas as pd

    # Two cycles, different lengths, 3 segments
    rng = np.random.default_rng(0)
    flex1 = rng.normal(size=(3, 10))
    flex2 = rng.normal(size=(3, 15))
    ext1 = rng.normal(size=(3, 9))
    ext2 = rng.normal(size=(3, 12))
    cd = CycleData(flex_mats=[flex1, flex2], ext_mats=[ext1, ext2])
    regions = pd.DataFrame({"Region": ["JC", "OT", "SB"], "Start": [1, 2, 3], "End": [1, 2, 3]})

    mf, me = compute_metric_averaged_metric(
        metric="total",
        cycle_data=cd,
        anatomical_regions=regions,
        n_interp_samples=20,
    )
    assert mf.shape == (20,)
    assert me.shape == (20,)

