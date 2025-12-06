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

    video = io.load_nparray("../data/segmented/1339_knee_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/segmented/1339_knee_radial_masks_N16.npy")
    radial_regions = io.load_nparray("../data/segmented/1339_knee_radial_regions_N16.npy")
    frm_offset = 289

    # Get pixel intensity sums
    lft = (11,1)
    mdl = (7,11)
    rgt = (1,7)
    total_sums = analyze_video(video, radial_masks, radial_regions, lft, mdl, rgt)



    # Plot specific cycles
    plot_specific_frames(total_sums, (290, 309), (312, 329), title="1339 - Total intensities - Cycle 1", start_frame=frm_offset)
    plot_specific_frames(total_sums, (331, 352), (355, 374), title="1339 - Total intensities - Cycle 2", start_frame=frm_offset)
    plot_specific_frames(total_sums, (375, 394), (398, 421), title="1339 - Total intensities - Cycle 3", start_frame=frm_offset)
    plot_specific_frames(total_sums, (422, 439), (441, 463), title="1339 - Total intensities - Cycle 4", start_frame=frm_offset)
    plot_specific_frames(total_sums, (464, 488), (490, 512), title="1339 - Total intensities - Cycle 5", start_frame=frm_offset)
    plot_specific_frames(total_sums, (513, 530), (532, 553), title="1339 - Total intensities - Cycle 6", start_frame=frm_offset)
    plot_specific_frames(total_sums, (554, 576), (579, 609), title="1339 - Total intensities - Cycle 7", start_frame=frm_offset)


    return


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
    # io.save_avi("../figures/1339_radial_segmentation.avi", v_out)
    # io.save_mp4("../figures/1339_radial_segmentation.mp4", v_out)

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1 + frm_offset, nfs + 1 + frm_offset)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    # df.to_excel("../figures/1339_radial_intensities.xlsx", index=False)

if __name__ == "__main__":
    main()