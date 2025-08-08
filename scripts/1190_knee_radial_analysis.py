import core.radial_segmentation as rdl
import core.data_processing as dp
import utils.views as views
import matplotlib.pyplot as plt
import utils.io as io
import numpy as np
import pandas as pd


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


def main():

    video = io.load_nparray("../data/processed/1190_knee_radial_video_N16.npy")
    radial_masks = io.load_nparray("../data/processed/1190_knee_radial_masks_N16.npy")
    radial_regions = io.load_nparray("../data/processed/1190_knee_radial_regions_N16.npy")

    print(video.shape, radial_masks.shape, radial_regions.shape)

    radial_masks = (radial_masks > 0).astype(np.uint8)*255


    lft = (11,1)
    mdl = (8,11)
    rgt = (1,8)

    l_knee = rdl.circular_slice(radial_masks, lft)
    l_knee = np.max(l_knee, axis=0)
    l_knee = np.minimum(l_knee, video)

    m_knee = rdl.circular_slice(radial_masks, mdl)
    m_knee = np.max(m_knee, axis=0)
    m_knee = np.minimum(m_knee, video)

    r_knee = rdl.circular_slice(radial_masks, rgt)
    r_knee = np.max(r_knee, axis=0)
    r_knee = np.minimum(r_knee, video)

    views.draw_radial_masks(video, [l_knee, m_knee, r_knee])

    total_sums = dp.measure_radial_intensities([l_knee, m_knee, r_knee])
    print(total_sums.shape) # Shape (3, 988)


    # Plot figures
    plot_with_xaxis_break(total_sums, 120, 400)

    return

    # Get pixel intensity sums
    lft = (11,1)
    mdl = (7,11)
    rgt = (1,7)
    total_sums = analyze_video(video, radial_regions, lft, mdl, rgt)

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
                    554, 576, 579, 609]) - 1 # -1 for 0-based indexing
    for i in range(0, len(lns), 2):
        plt.axvspan(lns[i], lns[i+1], color='gray', alpha=0.2)

    for i in range(0, len(lns), 4):
        mid = (lns[i] + lns[i+3]) / 2
        plt.text(mid, plt.ylim()[1] * 0.98, f"Cycle {i//4 + 1}",
             ha='center', va='top', fontsize=12, color='black')

    plt.title("1339 knee total pixel intensities (frames 290-609)")
    plt.legend()

    xticks = plt.xticks()[0]
    plt.xticks(xticks, [str(int(x+1)) for x in xticks]) # Display as 1-based indexing
    plt.show()

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