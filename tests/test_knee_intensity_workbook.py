"""Tests for src.utils.io workbook loading."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.io import (
    load_knee_intensity_workbook,
    SHEET_SEGMENT_INTENSITIES,
    SHEET_SEGMENT_INTENSITIES_BGSUB,
    SHEET_NUM_PIXELS,
    SHEET_MERGED_CYCLES,
    SHEET_FLEX_LEGACY,
    SHEET_EXT_LEGACY,
    SHEET_ANATOMICAL_REGIONS,
)


def _write_workbook(
    path: Path,
    *,
    include_bgsub: bool = True,
    include_pixels: bool = True,
    include_regions: bool = True,
    cycle_format: str = "merged",
) -> None:
    frames = pd.DataFrame(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        index=[1, 2, 3],
        columns=["Segment 1", "Segment 2"],
    )
    bgsub = pd.DataFrame(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        index=[1, 2, 3],
        columns=["Segment 1", "Segment 2"],
    )
    pixels = pd.DataFrame(
        [[5, 10], [5, 10], [5, 10]],
        index=[1, 2, 3],
        columns=["Segment 1", "Segment 2"],
    )
    regions = pd.DataFrame(
        {"Region": ["JC", "SB"], "Start": [1, 2], "End": [1, 2]}
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frames.to_excel(writer, sheet_name=SHEET_SEGMENT_INTENSITIES)
        if include_bgsub:
            bgsub.to_excel(writer, sheet_name=SHEET_SEGMENT_INTENSITIES_BGSUB)
        if include_pixels:
            pixels.to_excel(writer, sheet_name=SHEET_NUM_PIXELS)
        if include_regions:
            regions.to_excel(writer, sheet_name=SHEET_ANATOMICAL_REGIONS, index=False)

        if cycle_format == "merged":
            merged = pd.DataFrame(
                [
                    ["Flx Start", "Flx End", "Ext Start", "Ext End"],
                    [1, 2, 2, 3],
                ]
            )
            merged.to_excel(writer, sheet_name=SHEET_MERGED_CYCLES, header=False, index=False)
        elif cycle_format == "legacy":
            flex = pd.DataFrame(
                [
                    ["Cycle", "Start", "End"],
                    [1, 1, 2],
                ]
            )
            ext = pd.DataFrame(
                [
                    ["Cycle", "Start", "End"],
                    [1, 2, 3],
                ]
            )
            flex.to_excel(writer, sheet_name=SHEET_FLEX_LEGACY, header=False, index=False)
            ext.to_excel(writer, sheet_name=SHEET_EXT_LEGACY, header=False, index=False)
        else:
            raise ValueError(cycle_format)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_loads_raw_and_merged_cycles(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path, cycle_format="merged")

    wb = load_knee_intensity_workbook(path, source="raw")

    assert wb.intensities.shape == (3, 2)
    assert wb.num_pixels is None
    assert wb.anatomical_regions is None
    assert list(wb.flexion_cycles.columns) == ["start", "end"]
    assert list(wb.extension_cycles.columns) == ["start", "end"]
    assert wb.flexion_cycles.iloc[0].to_dict() == {"start": 1, "end": 2}
    assert wb.extension_cycles.iloc[0].to_dict() == {"start": 2, "end": 3}


def test_loads_bgsub_sheet(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path)

    wb = load_knee_intensity_workbook(path, source="bgsub")

    np.testing.assert_allclose(wb.intensities, [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])


def test_loads_optional_pixels_and_regions(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path, include_pixels=True, include_regions=True)

    wb = load_knee_intensity_workbook(
        path,
        source="raw",
        include_pixels=True,
        include_regions=True,
    )

    assert wb.num_pixels is not None
    assert wb.num_pixels.shape == (3, 2)
    assert wb.anatomical_regions is not None
    assert list(wb.anatomical_regions.columns) == ["Region", "Start", "End"]


def test_falls_back_to_legacy_cycle_sheets(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path, cycle_format="legacy")

    wb = load_knee_intensity_workbook(path, source="raw")

    assert wb.flexion_cycles.iloc[0].to_dict() == {"start": 1, "end": 2}
    assert wb.extension_cycles.iloc[0].to_dict() == {"start": 2, "end": 3}


def test_raises_when_cycle_sheets_missing(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    frames = pd.DataFrame([[1, 2]], index=[1], columns=["Segment 1", "Segment 2"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frames.to_excel(writer, sheet_name=SHEET_SEGMENT_INTENSITIES)

    with pytest.raises(FileNotFoundError):
        load_knee_intensity_workbook(path, source="raw")


def test_raises_when_pixels_requested_but_missing(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path, include_pixels=False)

    with pytest.raises(FileNotFoundError):
        load_knee_intensity_workbook(path, source="raw", include_pixels=True)


def test_raises_when_regions_requested_but_missing(tmp_path: Path) -> None:
    path = tmp_path / "test.xlsx"
    _write_workbook(path, include_regions=False)

    with pytest.raises(FileNotFoundError):
        load_knee_intensity_workbook(path, source="raw", include_regions=True)


# ---------------------------------------------------------------------------
# Integration test — uses real workbook from data/
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_workbook_smoke() -> None:
    path = Path("data/intensities_total/0000N64intensities.xlsx")
    if not path.exists():
        pytest.skip("real workbook not available")

    wb = load_knee_intensity_workbook(
        path,
        source="raw",
        include_pixels=True,
        include_regions=True,
    )

    assert wb.intensities.ndim == 2
    assert wb.intensities.shape[0] > 0
    assert wb.intensities.shape[1] > 0
    assert len(wb.flexion_cycles) == len(wb.extension_cycles)
    assert wb.num_pixels is not None
    assert wb.num_pixels.shape == wb.intensities.shape
    assert wb.anatomical_regions is not None
    assert wb.path == path