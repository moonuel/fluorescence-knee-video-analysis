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


def plot_with_xaxis_break(total_sums, skip_start, skip_end):
    cols = ['r', 'g', 'b']
    lbls = ["Left", "Middle", "Right"]

    # Original frames range
    full_start = 65
    full_end = 861

    # Shifted frame counts after removing skip range
    skip_length = skip_end - skip_start + 1
    new_length = (full_end - full_start + 1) - skip_length

    # Prepare new x axis that "skips" frames between skip_start and skip_end
    # We'll map frames [full_start..skip_start-1] to [0..skip_start - full_start -1]
    # and frames [skip_end+1..full_end] shifted left by skip_length
    
    def frame_to_new_x(f):
        if f < skip_start:
            return f - full_start
        elif f > skip_end:
            return f - full_start - skip_length
        else:
            return None  # frames in skip range are omitted
    
    # Prepare masked x indices and data
    x_vals = []
    y_vals = []
    nslices = total_sums.shape[0]

    for slc in range(nslices):
        x = []
        y = []
        for i in range(full_start, full_end + 1):
            new_x = frame_to_new_x(i)
            if new_x is not None:
                x.append(new_x)
                y.append(total_sums[slc, i])
        x_vals.append(np.array(x))
        y_vals.append(np.array(y))

    # Plotting
    plt.figure(figsize=(17,9))

    for slc in range(nslices):
        plt.plot(x_vals[slc], y_vals[slc], color=cols[slc], label=lbls[slc])

    # Shade regions - need to remap lns indices similarly
    lns = np.array([
        66, 89, 92, 109,
        421, 452, 470, 492,
        503, 532, 533, 569,
        737, 767, 770, 793,
        794, 822, 823, 860
    ]) - 1  # 0-based indexing

    def remap_frame(f):
        if f < skip_start:
            return f - full_start
        elif f > skip_end:
            return f - full_start - skip_length
        else:
            return None

    for i in range(0, len(lns), 2):
        start = remap_frame(lns[i])
        end = remap_frame(lns[i+1])
        if start is not None and end is not None:
            plt.axvspan(start, end, color='gray', alpha=0.2)

    for i in range(0, len(lns), 4):
        left = remap_frame(lns[i])
        right = remap_frame(lns[i+3])
        if left is not None and right is not None:
            mid = (left + right) / 2
            plt.text(mid, plt.ylim()[1] * 0.98, f"Cycle {i//4 + 1}",
                     ha='center', va='top', fontsize=12, color='black')

    plt.title(f"1190 knee total pixel intensities (frames {full_start}-{full_end} with frames {skip_start}-{skip_end} skipped)")
    plt.legend()

    # Fix x ticks: label them with the original frame numbers that remain visible
    new_xticks = plt.xticks()[0]
    labels = []
    for t in new_xticks:
        # invert frame_to_new_x mapping to label ticks correctly
        # since mapping is piecewise linear, invert:
        if t < skip_start - full_start:
            labels.append(str(int(t + full_start)))
        else:
            labels.append(str(int(t + full_start + skip_length)))

    plt.xticks(new_xticks, labels)

    # Draw the zigzag break on the x axis
    # Let's draw two small diagonal lines where the break happens
    break_pos = skip_start - full_start - 0.5  # just before the gap

    kwargs = dict(transform=plt.gca().get_xaxis_transform(), color='k', clip_on=False, linewidth=1.5)
    plt.plot([break_pos - 0.1, break_pos + 0.1], [-0.02, 0.02], **kwargs)
    plt.plot([break_pos - 0.1, break_pos + 0.1], [0.02, -0.02], **kwargs)

    plt.xlabel("Frame number")
    plt.ylabel("Total pixel intensity")
    plt.show()


def plot_with_multiple_xaxis_breaks(total_sums, skip_ranges, full_start=65, full_end=861):
    """
    Plot total_sums while skipping multiple frame ranges on the x-axis with breaks.
    
    Parameters:
        total_sums : np.array, shape (nslices, n_frames)
            The y-data for plotting.
        skip_ranges : list of tuples [(start1, end1), (start2, end2), ...]
            List of frame ranges to skip (inclusive).
        full_start : int
            Starting frame index of the full data range.
        full_end : int
            Ending frame index of the full data range.
    """
    cols = ['r', 'g', 'b']
    lbls = ["Left", "Middle", "Right"]
    
    # Sort and merge overlapping/adjacent skip ranges
    skip_ranges = sorted(skip_ranges, key=lambda x: x[0])
    merged_skips = []
    for rng in skip_ranges:
        if not merged_skips:
            merged_skips.append(rng)
        else:
            last_start, last_end = merged_skips[-1]
            curr_start, curr_end = rng
            if curr_start <= last_end + 1:  # overlapping or adjacent
                merged_skips[-1] = (last_start, max(last_end, curr_end))
            else:
                merged_skips.append(rng)
    skip_ranges = merged_skips
    
    # Precompute cumulative skip lengths and boundaries
    skip_lengths = [end - start + 1 for start, end in skip_ranges]
    cum_skip_lengths = np.cumsum(skip_lengths)
    
    # Function to compute how many frames are skipped before a given frame f
    def total_skip_before(f):
        total = 0
        for i, (start, end) in enumerate(skip_ranges):
            if f > end:
                total += skip_lengths[i]
            elif f >= start:
                # Inside a skip range → treated as skipped frame
                return None
            else:
                break
        return total
    
    # Map original frame f to compressed x-axis coordinate
    def frame_to_new_x(f):
        skip_before = total_skip_before(f)
        if skip_before is None:
            return None  # skip this frame
        return f - full_start - skip_before
    
    # Prepare masked x and y for each slice
    nslices = total_sums.shape[0]
    x_vals = []
    y_vals = []
    
    for slc in range(nslices):
        x = []
        y = []
        for i in range(full_start, full_end + 1):
            new_x = frame_to_new_x(i)
            if new_x is not None:
                x.append(new_x)
                y.append(total_sums[slc, i])
        x_vals.append(np.array(x))
        y_vals.append(np.array(y))
    
    plt.figure(figsize=(17, 9))
    
    # Plot data lines
    for slc in range(nslices):
        plt.plot(x_vals[slc], y_vals[slc], color=cols[slc], label=lbls[slc])
    
    # Shade regions - remap lns indices
    lns = np.array([
        66, 89, 92, 109,
        421, 452, 470, 492,
        503, 532, 533, 569,
        737, 767, 770, 793,
        794, 822, 823, 860
    ]) - 1  # zero-based
    
    def remap_frame(f):
        return frame_to_new_x(f)
    
    for i in range(0, len(lns), 2):
        start = remap_frame(lns[i])
        end = remap_frame(lns[i + 1])
        if start is not None and end is not None:
            plt.axvspan(start, end, color='gray', alpha=0.2)
    
    for i in range(0, len(lns), 4):
        left = remap_frame(lns[i])
        right = remap_frame(lns[i + 3])
        if left is not None and right is not None:
            mid = (left + right) / 2
            plt.text(mid, plt.ylim()[1] * 0.98, f"Cycle {i//4 + 1}",
                     ha='center', va='top', fontsize=12, color='black')
    
    plt.title(f"1190 knee total pixel intensities (frames {full_start}-{full_end} with skips)")
    plt.legend()
    
    # Set xticks and labels with correct mapping back to original frame numbers
    new_xticks = plt.xticks()[0]
    labels = []
    for t in new_xticks:
        # Invert frame_to_new_x mapping
        # This is more involved with multiple skip ranges:
        # Find original frame f so that frame_to_new_x(f) == t
        # We do a search since the mapping is piecewise linear
        # A binary search over frames is efficient
        low = full_start
        high = full_end
        orig_f = None
        while low <= high:
            mid = (low + high) // 2
            mapped = frame_to_new_x(mid)
            if mapped is None or mapped < t:
                low = mid + 1
            elif mapped > t:
                high = mid - 1
            else:
                orig_f = mid
                break
        if orig_f is None:
            # If exact match not found, approximate by closest smaller frame
            # Try low-1 first
            for candidate in range(low - 1, full_start - 1, -1):
                if frame_to_new_x(candidate) == t:
                    orig_f = candidate
                    break
            else:
                # fallback
                orig_f = int(t + full_start)
        labels.append(str(orig_f + 1))  # back to 1-based indexing
    
    plt.xticks(new_xticks, labels)
    plt.xlabel("Frame number")
    plt.ylabel("Total pixel intensity")
    
    # Draw zigzag breaks for each skip range
    ax = plt.gca()
    trans = ax.get_xaxis_transform()
    kwargs = dict(transform=trans, color='k', clip_on=False, linewidth=1.5)
    for start, end in skip_ranges:
        # Draw break *just before* where the gap starts
        break_pos = frame_to_new_x(start)
        if break_pos is None:
            # The break is at the end of previous frames: find next valid position left of start
            # Scan backwards until we find a valid frame
            for f in range(start - 1, full_start - 1, -1):
                pos = frame_to_new_x(f)
                if pos is not None:
                    break_pos = pos + 0.5  # put break slightly right of last valid frame
                    break
            else:
                continue  # no valid frame found, skip break
        plt.plot([break_pos - 0.1, break_pos + 0.1], [-0.02, 0.02], **kwargs)
        plt.plot([break_pos - 0.1, break_pos + 0.1], [0.02, -0.02], **kwargs)
    
    plt.show()


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


def main():

    video = io.load_nparray("../data/processed/1193_normal_radial_video_N16.npy")#[1700:1750]
    radial_masks = io.load_nparray("../data/processed/1193_normal_radial_masks_N16.npy")#[1700:1750]

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

    l_knee = radial_masks[np.isin(radial_masks, [1,2,3,4,5,6])] = 1
    m_knee = radial_masks[np.isin(radial_masks, [7,8,9])] = 2
    r_knee = radial_masks[np.isin(radial_masks, [10,11,12,13,14,15,16])] = 3

    video_w_bnds = draw_mask_boundaries(video, radial_masks)
    views.show_frames(video_w_bnds)

    

    return

    print(video.shape, radial_masks.shape)

    radial_masks = (radial_masks > 0).astype(np.uint8)*255


    lft = (11,1)
    mdl = (8,11)
    rgt = (1,8)

    l_knee = rdl.circular_slice(radial_masks, lft)
    l_knee = np.max(l_knee, axis=0) # combine the masks
    l_knee = np.minimum(l_knee, video) # restrict video to mask

    m_knee = rdl.circular_slice(radial_masks, mdl)
    m_knee = np.max(m_knee, axis=0) # combine the masks
    m_knee = np.minimum(m_knee, video) # restrict video to mask

    r_knee = rdl.circular_slice(radial_masks, rgt)
    r_knee = np.max(r_knee, axis=0) # combine the masks
    r_knee = np.minimum(r_knee, video) # restrict video to mask

    v_out = views.draw_radial_masks(video, [l_knee, m_knee, r_knee])

    total_sums = dp.measure_radial_intensities([l_knee, m_knee, r_knee])
    print(total_sums.shape) # Shape (3, 988)


    # Plot figures
    # plot_with_xaxis_break(total_sums, 120, 400)
    plot_with_multiple_xaxis_breaks(total_sums, [(120, 400), (590, 720)])

    plot_specific_frames(total_sums, 
                         flex_frames=(66, 89),
                         ext_frames=(92, 109),
                         title = "1190 - Total pixel intensities - Cycle 1")
    
    plot_specific_frames(total_sums, 
                         flex_frames=(421, 452),
                         ext_frames=(470, 492),
                         title = "1190 - Total pixel intensities - Cycle 2")
    
    plot_specific_frames(total_sums, 
                         flex_frames=(503, 532),
                         ext_frames=(533, 569),
                         title = "1190 - Total pixel intensities - Cycle 3")
    
    plot_specific_frames(total_sums, 
                         flex_frames=(737, 767),
                         ext_frames=(770, 793),
                         title = "1190 - Total pixel intensities - Cycle 4")

    plot_specific_frames(total_sums, 
                         flex_frames=(794, 822),
                         ext_frames=(823, 860),
                         title = "1190 - Total pixel intensities - Cycle 5")

    # io.save_avi("../figures/1190 knee radial analysis/1190_radial_segmentation.avi", v_out, fps=60)
    # io.save_mp4("../figures/1190 knee radial analysis/1190_radial_segmentation.mp4", v_out, fps=60)

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1, nfs + 1)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    # df.to_excel("../figures/1190 knee radial analysis/1190_radial_intensities.xlsx", index=False)

    return

    # Show video for validation
    r_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, lft))
    r_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, mdl))
    r_knee = rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    v_out = views.draw_radial_masks(video, [r_knee, r_knee, r_knee], frame_offset=290)
    io.save_avi("../figures/1339_radial_segmentation.avi", v_out)
    io.save_mp4("../figures/1339_radial_segmentation.mp4", v_out)

    # Write data to spreadsheet
    nfs = total_sums.shape[1]
    fns = np.arange(1, nfs + 1)
    df = pd.DataFrame(np.column_stack([fns, total_sums.T]), columns=["Frame Number", "Left", "Middle", "Right"])

    df.to_excel("../figures/1339_radial_intensities.xlsx", index=False)

if __name__ == "__main__":
    main()