import os
import sys
from src.core import knee_segmentation as ks
from src.core import data_processing as dp
from src.utils import io, views
import matplotlib.pyplot as plt

def main():
    print(__name__ + " called!")

    # Process the video data 
    # video = io.load_tif("../data/1 aging_00000221.tif") 
    # video_ctrd, translation_mxs = ks.pre_process_video(video) # Centers *all* frames
    video_ctrd = io.load_nparray("../data/processed/aging_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/aging_translation_mxs.npy")

    # Process the coord data. TODO: wrap coords-dependent code in a loop to process all data sets at once?
    knee_name = "aging-3" 
    coords, metadata = io.load_aging_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", knee_name)
    coords_ctrd = ks.translate_coords(translation_mxs, coords) # Processes *some* frames
    coords_ctrd = ks.smooth_coords(coords_ctrd, 5) # Implements a moving average filter over the coordinate data
    # views.plot_coords(video_ctrd, coords_ctrd) # Validate smoothing

    # Get masks
    regions, masks = ks.get_three_segments(video_ctrd, coords_ctrd)  
    # views.display_regions(regions, keys) # Validate regions

    # Get intensity data
    keys = ["l", "m", "r"]
    raw_intensities = dp.measure_region_intensities(regions, masks, keys) # Returns a dict
    normalized_intensities = dp.measure_region_intensities(regions, masks, keys, normalized=True)

    # Plot intensities
    show_figs = False
    save_figs = True
    figsize=(10,15)
    views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)
    views.plot_three_intensities(normalized_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)

    # exit(0)

    # Get per-region rate of change
    raw_deriv = dp.get_intensity_diffs(raw_intensities)
    # raw_deriv = dp.get_intensity_derivs(raw_intensities) # second order accuracy

    # Plot derivatives 
    views.plot_three_derivs(raw_deriv, metadata, show_figs, save_figs, figsize=figsize)
    
    

if __name__ == "__main__":
    main()

# TODO: refine Dict[] type hints where there are multiple return types? consider making an object class hmmmm 