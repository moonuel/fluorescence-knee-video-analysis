import numpy as np
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

def analyze_video(video, radial_masks, radial_regions, 
                  lft: Tuple[int, int], mdl: Tuple[int, int], rgt: Tuple[int, int], 
                  show_figs: bool = True, save_dir: str = None, 
                  fig_size: Tuple[int, int] = (17, 9)) -> Tuple[np.ndarray, matplotlib.figure.Figure, plt.Axes]:
    """Analyzes all frames in a radially-segmented knee fluorescence video.

    Parameters
    ----------
    video : np.ndarray
        Input video of shape (n_frames, H, W)
    radial_masks : np.ndarray
        Binary mask array of shape (n_slices, n_frames, H, W)
    radial_regions : np.ndarray
        Binary region array of same shape as radial_masks
    lft, mdl, rgt : Tuple[int, int]
        Circular slice ranges for left/middle/right knees
    show_figs : bool, optional
        Whether to display the figure
    save_dir : str, optional
        Directory to save output figure. If None, the figure is not saved.
    fig_size : Tuple[int, int], optional
        Size of the matplotlib figure

    Returns
    -------
    total_sums : np.ndarray
        Measured intensities
    fig : matplotlib.figure.Figure
        The generated figure
    axes : np.ndarray
        The figure's axes
    """
    if VERBOSE: print("analyze_video() called!")

    video = video.copy()
    nfs, h, w = video.shape

    assert nfs == radial_masks.shape[1] 
    assert nfs == radial_regions.shape[1]

    masks = {
        'l': rdl.combine_masks(rdl.circular_slice(radial_masks, lft)), 
        'm': rdl.combine_masks(rdl.circular_slice(radial_masks, mdl)),
        'r': rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    }
    
    regions = {
        'l': rdl.combine_masks(rdl.circular_slice(radial_regions, lft)), 
        'm': rdl.combine_masks(rdl.circular_slice(radial_regions, mdl)),
        'r': rdl.combine_masks(rdl.circular_slice(radial_regions, rgt))
    }
    
    keys = ['l','m','r']
    
    total_sums = dp.measure_radial_intensities(np.asarray([
        regions["l"], regions["m"], regions["r"]
    ]))

    fig = plot_radial_segment_intensities_combined(total_sums, vert_layout=True, show_figs=show_figs, save_dir=save_dir, figsize=(17,9))

    if show_figs:
        plt.show()

    return total_sums, fig

def main():
    if VERBOSE: print("main() called!")

    video = io.load_nparray("../data/processed/normal_knee_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/processed/normal_knee_radial_masks_N16.npy")
    radial_regions = io.load_nparray("../data/processed/normal_knee_radial_regions_N16.npy")

    lft = (14,2)
    mdl = (9,14)
    rgt = (2,9)

    save_dir = "../docs/meetings/16-Jul-2025/normal_intensity_plots/"
    save_dir = None 
    total_sums, figs = analyze_video(video, radial_masks, radial_regions, lft, mdl, rgt, 
                                           show_figs=False, save_dir=save_dir)
    

    plt.axvline(71, color='b', linestyle="-")

    plt.show()

    exit(420)

    lft_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, lft))
    mdl_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, mdl))
    rgt_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    v_out = views.draw_radial_masks(video, [lft_mask, mdl_mask, rgt_mask])

    return

if __name__ == "__main__":
    main()