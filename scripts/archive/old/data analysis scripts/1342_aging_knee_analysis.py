import core.radial_segmentation as rdl
import core.data_processing as dp
import utils.views as views
import matplotlib.pyplot as plt
import utils.io as io
import numpy as np
import pandas as pd
import cv2
from config import VERBOSE
from typing import Tuple


def analyze_video(video, radial_regions, 
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

    # assert nfs == radial_masks.shape[1] 
    assert nfs == radial_regions.shape[1]

    # masks = {
    #     'l': rdl.combine_masks(rdl.circular_slice(radial_masks, lft)), 
    #     'm': rdl.combine_masks(rdl.circular_slice(radial_masks, mdl)),
    #     'r': rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    # }
    
    views.show_frames(radial_regions[0])

    regions = {
        'l': rdl.combine_masks(rdl.circular_slice(radial_regions, lft)), 
        'm': rdl.combine_masks(rdl.circular_slice(radial_regions, mdl)),
        'r': rdl.combine_masks(rdl.circular_slice(radial_regions, rgt))
    }

    views.show_frames(regions["l"]) # Validate results
    views.show_frames(regions["m"])
    views.show_frames(regions["r"])

    
    total_sums = dp.measure_radial_intensities(np.asarray([
        regions["l"], regions["m"], regions["r"]
    ]))

    return total_sums


def plot_specific_frames(total_sums, flex_frames=None, ext_frames=None, title=""):
    """
    Plot total_sums curves with shaded frame regions.

    Parameters:
    - total_sums: np.ndarray, shape (3, nframes)
    - flex_frames: tuple (start_frame, end_frame), frames to shade gray
    - ext_frames: tuple (start_frame, end_frame), frames to shade gray
    - title: str, plot title
    """
    import matplotlib.ticker as ticker

    nframes = total_sums.shape[1]
    cols = ['r', 'g', 'b']
    lbls = ["Left", "Middle", "Right"]

    plt.figure(figsize=(9, 17))
    plt.xlim(1, nframes)

    x = np.arange(1, nframes + 1)  # 1-based frame indices

    # Plot each curve
    for slc in range(3):
        plt.plot(x, total_sums[slc], color=cols[slc], label=lbls[slc])

    # Shade flex_frames if provided
    if flex_frames is not None:
        plt.axvspan(flex_frames[0], flex_frames[1], color='gray', alpha=0.3)
        plt.axvline(flex_frames[1], color = 'k', linestyle = '--')
        plt.xlim(left=flex_frames[0])
        mid_flex = (flex_frames[0] + flex_frames[1]) / 2
        plt.text(mid_flex, plt.ylim()[1] * 0.98, "Flexion",
                 ha='center', va='top', fontsize=12, color='black')

    # Shade ext_frames if provided
    if ext_frames is not None:
        plt.axvspan(ext_frames[0], ext_frames[1], color='gray', alpha=0.3)
        plt.axvline(ext_frames[0], color = 'k', linestyle = '--')
        plt.xlim(right=ext_frames[1])
        mid_ext = (ext_frames[0] + ext_frames[1]) / 2
        plt.text(mid_ext, plt.ylim()[1] * 0.98, "Extension",
                 ha='center', va='top', fontsize=12, color='black')

    plt.title(title)
    plt.legend()
    plt.xlabel("Frame number")
    plt.ylabel("Total pixel intensity")

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.show()
    
    return


def sum_intensity_per_partition(video: np.ndarray, radial_masks: np.ndarray, N: int) -> np.ndarray:
    """
    Vectorized computation of sum of pixel intensities in each partition for every frame.

    Args:
        video: np.ndarray of shape (nfs, h, w), pixel intensities
        radial_masks: np.ndarray of shape (nfs, h, w), integer mask with values 0 (background) or 1..N
        N: int, number of partitions

    Returns:
        sums_per_frame: np.ndarray of shape (N, nfs)
                        sums_per_frame[i, cf] is the sum of pixels in partition i+1 of frame cf
        counts_per_frame: np.ndarray of shape (N, nfs)
                          counts_per_frame[i, cf] is the count of pixels in partition i+1 of frame cf of the video
    """
    nfs, h, w = video.shape
    sums_per_frame = np.zeros((N, nfs), dtype=np.uint32)
    counts_per_frame = np.zeros((N, nfs), dtype=np.uint32)

    for n in range(1, N+1):
        # Create a boolean mask for the current partition (1..N)
        mask_one_hot = (radial_masks == n)  # shape (nfs, h, w)

        # Multiply by video and sum over h and w axes
        sums_per_frame[n-1, :] = np.sum(video * mask_one_hot, axis=(1, 2))
        counts_per_frame[n-1, :] = np.sum(mask_one_hot, axis=(1, 2))

    return sums_per_frame, counts_per_frame


def draw_mask_boundaries(video: np.ndarray, mask_labels: np.ndarray, intensity: int = 255) -> np.ndarray:
    """
    Draws boundaries between partitions in mask_labels on grayscale video frames.
    Uses a fast vectorized method instead of per-label contours.

    Args:
        video (np.ndarray): Grayscale video of shape (nframes, h, w), dtype uint8.
        mask_labels (np.ndarray): Labeled mask array of shape (nframes, h, w),
                                  where 0 = background, 1..N = partitions.
        intensity (int): Pixel intensity for boundaries (0-255).

    Returns:
        np.ndarray: Video with boundaries drawn, shape (nframes, h, w), dtype uint8.
    """
    nframes, h, w = video.shape
    output = video.copy()

    for i in range(nframes):
        labels = mask_labels[i]

        # Compute adjacency differences
        horiz_diff = labels[:, 1:] != labels[:, :-1]
        vert_diff = labels[1:, :] != labels[:-1, :]

        # Create an edge map
        edges = np.zeros_like(labels, dtype=bool)
        edges[:, 1:] |= horiz_diff
        edges[1:, :] |= vert_diff

        # Draw edges on the frame
        output[i][edges] = intensity

    return output


def plot_with_shading(total_sums, lns, f0=None, fN=None):
    """
    Plot total_sums between frames f0 and fN with shaded regions and cycle labels.

    Parameters:
    - total_sums: np.ndarray, shape (nslices, nframes)
    - f0: int, starting frame index (absolute)
    - fN: int, ending frame index (absolute)
    - lns: list or array of absolute frame numbers (even length),
           where each pair defines a shaded region, and
           every two pairs define a cycle.
    """

    if f0 is None: f0 = 0
    if fN is None: fN = total_sums.shape[1]

    cols = ['r', 'g', 'b']
    lbls = ["Left", "Middle", "Right"]
    nslices = total_sums.shape[0]

    # Prepare x and y values
    x = np.arange(f0, fN)
    y_vals = [total_sums[slc, f0:fN] for slc in range(nslices)]

    # Plot curves
    plt.figure(figsize=(17, 9))
    for slc in range(nslices):
        plt.plot(x, y_vals[slc], color=cols[slc], label=lbls[slc])

    # Shade regions
    for i in range(0, len(lns), 2):
        start, end = lns[i], lns[i+1]
        if end < f0 or start > fN:
            continue  # skip if completely outside range
        start = max(start, f0)
        end   = min(end, fN)
        plt.axvspan(start, end, color='gray', alpha=0.2)

    # Label cycles
    for i in range(0, len(lns), 4):
        left, right = lns[i], lns[i+3]
        if left >= f0 and right <= fN:
            mid = (left + right) / 2
            plt.text(mid, plt.ylim()[1] * 0.98,
                     f"Cycle {i//4 + 1}",
                     ha='center', va='top', fontsize=12, color='black')

    # Final touches
    plt.xlabel("Frame number")
    plt.ylabel("Total pixel intensity")
    plt.title(f"Total pixel intensities (frames {f0+1}-{fN+1})")
    plt.xlim(f0, fN)
    plt.legend()
    plt.show()


def main():

    video = io.load_nparray("../data/processed/aging_1342_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/processed/aging_1342_radial_masks_N16.npy")

    # video_w_bnds = draw_mask_boundaries(video, radial_masks)
    # views.show_frames(video_w_bnds)

    print(video.shape, radial_masks.shape)
    print(video.dtype, radial_masks.dtype)

    nfs, h, w = video.shape

    N = 16
    
    # total_sums, total_counts = sum_intensity_per_partition(video, radial_masks, N=16)

    # views.show_frames([np.isin(radial_masks, [1,2,3,4,5,6]) * video, video], "Left knee")
    # views.show_frames([np.isin(radial_masks, [7,8,9]) * video, video], "Middle knee")
    # views.show_frames([np.isin(radial_masks, [10,11,12,13,14,15,16]) * video, video], "Right knee")

    l_knee = radial_masks[np.isin(radial_masks, [1,2,3,4,5])] = 1
    m_knee = radial_masks[np.isin(radial_masks, [6,7,8,9])] = 2
    r_knee = radial_masks[np.isin(radial_masks, [10,11,12,13,14,15,16])] = 3

    video_w_bnds = draw_mask_boundaries(video, radial_masks)
    video_w_bnds = views.show_frames(video_w_bnds)

    total_sums, total_counts = sum_intensity_per_partition(video, radial_masks, 3)
    print(total_sums)
    print(total_sums.shape)


    lns = [
        62,81, 82,100,
        102,119, 123,151,
        152,171, 178,199,
        206,222, 223,246,
        247,272, 273,297,
        298,320, 321,340,
        341,364, 365,384
    ]
    plot_with_shading(total_sums, lns, 50)

    plot_specific_frames(total_sums, (62,81), (82,100), "1342 - Total pixel intensities - Cycle 1")
    plot_specific_frames(total_sums, (102,119), (123,151), "1342 - Total pixel intensities - Cycle 2")
    plot_specific_frames(total_sums, (152,171), (178,199), "1342 - Total pixel intensities - Cycle 3")
    plot_specific_frames(total_sums, (206,222), (223,246), "1342 - Total pixel intensities - Cycle 4")
    plot_specific_frames(total_sums, (247,272), (273,297), "1342 - Total pixel intensities - Cycle 5")
    plot_specific_frames(total_sums, (298,320), (321,340), "1342 - Total pixel intensities - Cycle 6")
    plot_specific_frames(total_sums, (341,364), (365,384), "1342 - Total pixel intensities - Cycle 7")

    io.save_avi("../figures/1342 knee radial analysis/1342_radial_segmentation.avi", video_w_bnds, fps=60)
    io.save_mp4("../figures/1342 knee radial analysis/1342_radial_segmentation.mp4", video_w_bnds, fps=60)

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1, nfs + 1)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    df.to_excel("../figures/1342 knee radial analysis/1342_radial_intensities.xlsx", index=False)

    return


if __name__ == "__main__":
    main()