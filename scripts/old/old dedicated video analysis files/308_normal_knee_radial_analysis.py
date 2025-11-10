import numpy as np
import pandas as pd
import cv2
import utils.io as io
import utils.views as views
import core.radial_segmentation as rdl
import core.data_processing as dp
from config import VERBOSE
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.figure
from pathlib import Path

def plot_radial_segment_intensities_combined(intensities: np.ndarray, f0: int = None, fN: int = None, 
                                      show_figs: bool = True, save_dir: str = None, 
                                      vert_layout: bool = False, figsize: Tuple[int, int] = (20, 7)) -> Tuple:
    """
    Different version so I don't break existing functions. Accepts 0-based frame indexing.
    Alternate function intended for use with the radial segmentation scheme.
    Plots all provided intensity figures. 

    Pass f0 = fN = None for default settings (whole video).

    Returns (fig, axes)
    
    Args:
        intensities (np.ndarray): shape (n_slices, n_frames), intensity data
        f0 (int): starting frame index (inclusive), defaults to 0
        fN (int): ending frame index (inclusive), defaults to last frame
        show_figs (bool): whether to show figures
        save_dir (str): if provided, saves the figure to this directory
        vert_layout (bool): vertical or horizontal subplots
        figsize (tuple): figure size
    """
    if VERBOSE: print("plot_radial_segment_intensities_2() called!")

    nslcs, nfrms = intensities.shape

    if f0 is None:
        f0 = 0
    if fN is None:
        fN = nfrms

    intensities = intensities[:, f0:fN].copy()
    frmns = np.arange(f0, fN)

    # === Plot all slices on same figure ===
    fig = plt.figure(figsize=figsize)
    for slc in range(nslcs):
        plt.plot(frmns, intensities[slc], label=f"Slice {slc}")
    plt.title("Radial Segment Intensities")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir).expanduser().resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        filepath_combined = save_path / "radial_intensity_combined.png"
        plt.savefig(filepath_combined, dpi=300, bbox_inches="tight")
        print(f"Saved combined intensity plot to {filepath_combined}")

    if show_figs:
        plt.show()
    else:
        plt.close()

    return fig

def sum_region_intensities(radial_regions, 
                           lft: Tuple[int, int], 
                           mdl: Tuple[int, int], 
                           rgt: Tuple[int, int]
                           ) -> Tuple[np.ndarray, matplotlib.figure.Figure, plt.Axes]:
    """Analyzes all frames in a radially-segmented 3 part knee fluorescence video.

    Parameters
    ----------
    radial_regions : np.ndarray
        Binary region array of same shape as radial_masks
    lft, mdl, rgt : Tuple[int, int]
        Circular slice ranges for left/middle/right knees

    Returns
    -------
    total_sums : np.ndarray
        Measured intensities
    """
    if VERBOSE: print("analyze_video() called!")

    l = rdl.combine_masks(rdl.circular_slice(radial_regions, lft))
    m = rdl.combine_masks(rdl.circular_slice(radial_regions, mdl))
    r = rdl.combine_masks(rdl.circular_slice(radial_regions, rgt))
    
    total_sums = dp.measure_radial_intensities(np.asarray([l,m,r]))

    return total_sums


def plot_specific_frames(total_sums, 
                         flex_frames=None, 
                         ext_frames=None, 
                         title="", 
                         start_frame=1):
    """
    Plot total_sums curves with shaded frame regions.

    Parameters:
    - total_sums: np.ndarray, shape (3, nframes)
    - flex_frames: tuple (start_frame, end_frame), frames to shade gray
    - ext_frames: tuple (start_frame, end_frame), frames to shade gray
    - title: str, plot title
    - start_frame: int, actual frame number of total_sums[*,0]
    """

    nframes = total_sums.shape[1]
    cols = ['r', 'g', 'b']
    lbls = ["Left", "Middle", "Right"]

    plt.figure(figsize=(9, 17))

    # Generate x-axis values based on start_frame
    x = np.arange(start_frame, start_frame + nframes)
    plt.xlim(start_frame, start_frame + nframes - 1)

    # Plot each curve
    for slc in range(3):
        plt.plot(x, total_sums[slc], color=cols[slc], label=lbls[slc])

    # Shade flex_frames if provided
    if flex_frames is not None:
        plt.axvspan(flex_frames[0], flex_frames[1], color='gray', alpha=0.3)
        plt.axvline(flex_frames[1], color='k', linestyle='--')
        mid_flex = (flex_frames[0] + flex_frames[1]) / 2
        plt.text(mid_flex, plt.ylim()[1] * 0.98, "Flexion",
                 ha='center', va='top', fontsize=12, color='black')
        plt.xlim(left=flex_frames[0])

    # Shade ext_frames if provided
    if ext_frames is not None:
        plt.axvspan(ext_frames[0], ext_frames[1], color='gray', alpha=0.3)
        plt.axvline(ext_frames[0], color='k', linestyle='--')
        mid_ext = (ext_frames[0] + ext_frames[1]) / 2
        plt.text(mid_ext, plt.ylim()[1] * 0.98, "Extension",
                 ha='center', va='top', fontsize=12, color='black')
        plt.xlim(right=ext_frames[1])

    plt.title(title)
    plt.legend()
    plt.xlabel("Frame number")
    plt.ylabel("Total pixel intensity")

    plt.show()
    return


def main():
    if VERBOSE: print("main() called!")

    video = io.load_nparray("../data/processed/normal_knee_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/processed/normal_knee_radial_masks_N16.npy")
    radial_regions = io.load_nparray("../data/processed/normal_knee_radial_regions_N16.npy")

    # Get total pixel intensity per slice
    rgt, mdl, lft = (2,9), (9,14), (14,2)

    save_dir = "../docs/meetings/16-Jul-2025/normal_intensity_plots/"
    save_dir = None # Disable save
    total_sums = sum_region_intensities(radial_regions, lft, mdl, rgt)
    
    plot_specific_frames(total_sums, (71, 116), (117, 155), "308 - Total intensities - Cycle 1")
    plot_specific_frames(total_sums, (253, 298), (299, 335), "308 - Total intensities - Cycle 2")
    plot_specific_frames(total_sums, (585, 618), (630, 669), "308 - Total intensities - Cycle 3")
    plot_specific_frames(total_sums, (156, 199), (210, 250), "308 - Total intensities - Cycle 4")

    return # Temp 

    # Plot figures
    cols = ['r', 'g', 'b'] # Hard code RGB colors for LMR
    lbls = ["Left", "Middle", "Right"]
    plt.figure(figsize=(17,9))
    nslices = total_sums.shape[0]
    for slc in range(nslices):
        plt.plot(total_sums[slc], color=cols[slc], label=lbls[slc])

    # Shade regions between line pairs
    lns = np.array([71, 116, 117, 155, 253, 298, 299, 335, 585, 618, 630, 669]) - 1 # -1 for 0-based indexing
    for i in range(0, len(lns), 2):
        cyc_n = i//2 + 1
        plt.axvspan(lns[i], lns[i+1], color='gray', alpha=0.2)

    for i in range(0, len(lns), 4):
        mid = (lns[i] + lns[i+3]) / 2
        plt.text(mid, plt.ylim()[1] * 0.98, f"Cycle {i//4 + 1}",
             ha='center', va='top', fontsize=12, color='black')

    plt.title("Normal (308) knee total pixel intensities")
    plt.legend()

    xticks = plt.xticks()[0]
    plt.xticks(xticks, [str(int(x+1)) for x in xticks]) # Display as 1-based indexing
    plt.show()

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1, nfs + 1)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    df.to_excel("../docs/meetings/16-Jul-2025/Normal_308_knee_total_intensities.xlsx", index=False)
    


    exit(420)

    lft_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, lft))
    mdl_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, mdl))
    rgt_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    v_out = views.draw_radial_masks(video, [lft_mask, mdl_mask, rgt_mask])

    return

if __name__ == "__main__":
    main()