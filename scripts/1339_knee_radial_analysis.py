import core.radial_segmentation as rdl
import core.data_processing as dp
import utils.views as views
import matplotlib.pyplot as plt
import utils.io as io
import numpy as np
import pandas as pd


from config import VERBOSE
from typing import Tuple

def analyze_video(video, radial_masks, radial_regions, 
                  lft: Tuple[int, int], mdl: Tuple[int, int], rgt: Tuple[int, int]) -> None:
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

    Returns
    -------
    total_sums : np.ndarray
        Measured intensities in shape (3, nfs), where total_sums[i] for i=0,1,2 is the left/middle/right knee respectively
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
    
    total_sums = dp.measure_radial_intensities(np.asarray([
        regions["l"], regions["m"], regions["r"]
    ]))

    return total_sums

def main():

    video = io.load_nparray("../data/processed/1339_knee_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/processed/1339_knee_radial_masks_N16.npy")
    radial_regions = io.load_nparray("../data/processed/1339_knee_radial_regions_N16.npy")
    frm_offset = 289

    # Get pixel intensity sums
    lft = (11,1)
    mdl = (7,11)
    rgt = (1,7)
    total_sums = analyze_video(video, radial_masks, radial_regions, lft, mdl, rgt)

    # Plot figures
    cols = ['r', 'g', 'b'] # Hard code RGB colors for LMR
    lbls = ["Left", "Middle", "Right"]

    plt.figure(figsize=(17,9))
    
    nslices = total_sums.shape[0]
    for slc in range(nslices):
        plt.plot(total_sums[slc], color=cols[slc], label=lbls[slc])

    # Shade regions between line pairs
    lns = np.array([290, 309, 312, 329, 
                    331, 352, 355, 374, 
                    375, 394, 398, 421, 
                    422, 439, 441, 463, 
                    464, 488, 490, 512, 
                    513, 530, 532, 553, 
                    554, 576, 579, 609]) - 1 - frm_offset # -1 for 0-based indexing
    for i in range(0, len(lns), 2):
        plt.axvspan(lns[i], lns[i+1], color='gray', alpha=0.2)

    for i in range(0, len(lns), 4):
        mid = (lns[i] + lns[i+3]) / 2
        plt.text(mid, plt.ylim()[1] * 0.98, f"Cycle {i//4 + 1}",
             ha='center', va='top', fontsize=12, color='black')

    plt.title("1339 knee total pixel intensities (frames 290-609)")
    plt.legend()

    xticks = plt.xticks()[0]
    plt.xticks(xticks, [str(int(x+1 + frm_offset)) for x in xticks]) # Display as 1-based indexing
    plt.show()

    # Show video for validation
    l_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, lft))
    m_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, mdl))
    r_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    v_out = views.draw_radial_masks(video, [l_knee, m_knee, r_knee], frame_offset=290)
    io.save_avi("../figures/1339_radial_segmentation.avi", v_out)
    io.save_mp4("../figures/1339_radial_segmentation.mp4", v_out)

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1 + frm_offset, nfs + 1 + frm_offset)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    df.to_excel("../figures/1339_radial_intensities.xlsx", index=False)

if __name__ == "__main__":
    main()