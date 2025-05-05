import os
import sys
from src.core import knee_segmentation as ks
from src.core import calculate_data as cd
from src.utils import io, views
import matplotlib.pyplot as plt

def main():
    print(__name__ + " called!")

    # Process the video data 
    video = io.load_tif("../data/1 aging_00000221.tif") # TODO: for performance gains: write the centered video data to file, but first check if it exists
    # video_ctrd, translation_mxs = ks.pre_process_video(video) # Centers *all* frames
    video_ctrd = io.load_nparray("../data/processed/aging_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/translation_mxs.npy")

    # Process the coord data. TODO: wrap coords-dependent code in a loop to process all data sets at once?
    knee_name = "aging-1" 
    coords, metadata = io.load_aging_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", knee_name)
    coords_ctrd = ks.translate_coords(translation_mxs, coords) # Processes *some* frames
    coords_ctrd = ks.smooth_coords(coords_ctrd, 5) # Implements a moving average filter over the coordinate data

    # views.plot_coords(video_ctrd, coords_ctrd) # Validate smoothing

    # exit(0)

    # Get masks
    regions, masks = ks.get_three_segments(video_ctrd, coords_ctrd)  
    keys = ["l", "m", "r"]

    views.display_regions(regions, keys) # Validate regions

    exit(0)
    
    # Get intensity data
    raw_intensities = cd.measure_region_intensities(regions, masks, keys) # Returns a dict
    normalized_intensities = cd.measure_region_intensities(regions, masks, keys, normalized=True)

    # Plot intensities
    show_figs = False
    save_figs = False
    views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs)
    views.plot_three_intensities(normalized_intensities, metadata, show_figs, save_figs)

    # exit(0)

    # Get per-region rate of change
    # raw_deriv = get_intensity_diffs(raw_intensities)
    raw_deriv = cd.get_intensity_derivs(raw_intensities) 

    plt.close('all')
    for k in keys:
        plt.plot(raw_deriv[k], label=f"{k} knee")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()

# TODO: refine Dict[] type hints where there are multiple return types? consider making an object class hmmmm 