import os
import sys
from core import knee_segmentation as ks
from core import data_processing as dp
from utils import io, views, utils
import matplotlib.pyplot as plt
import numpy as np

def main():
    print(__name__ + " called!")

    # Process the video data 
    # video = io.load_tif("../data/1 aging_00000221.tif") 
    # video_ctrd, translation_mxs = ks.centre_video(video) # Centers *all* frames
    video = io.load_nparray("../data/processed/aging_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/aging_translation_mxs.npy")

    knee_name = "aging-3" 
    coords, metadata = io.load_aging_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", knee_name)
    coords_ctrd = ks.translate_coords(translation_mxs, coords) # Processes *some* frames
    # coords_ctrd = ks.smooth_coords(coords_ctrd, 5) # Implements a moving average filter over the coordinate data
    # views.plot_coords(video_ctrd, coords_ctrd) # Validate smoothing

    # Transform video like Juan
    video = utils.blur_video(video, (11,11), sigma=3)
    # video = utils.log_transform_video(video, gain=1)

    # Get masks
    keys = ["l", "m", "r"]
    regions, masks = ks.get_three_segments(video, coords_ctrd, thresh_scale=1)  
    # views.show_regions(regions, keys) # Validate regions


    # Validate coords
    video_preview = views.plot_coords(video, coords_ctrd, show_video=False) # Plots coords on whole video. Not all frames have coords
    video_preview = utils.crop_video_square(video_preview, 350)
    views.show_frames(video_preview)
    # views.show_frames(np.max([regions['l'], regions['m'], regions['r']], axis=0)) # Validate mask

    exit(0)
    # Get intensity data
    raw_intensities = dp.measure_region_intensities(regions, masks, keys) # Returns a dict
    normalized_intensities = dp.measure_region_intensities(regions, masks, keys, normalized=True)

    # Plot intensities
    show_figs = True
    save_figs = True
    figsize=(9,17)
    views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)
    # views.plot_three_intensities(normalized_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)

    # exit(0)

    # Get per-region rate of change
    # raw_deriv = dp.get_intensity_diffs(raw_intensities)
    # raw_deriv = dp.get_intensity_derivs(raw_intensities) # second order accuracy

    # Plot derivatives 
    # views.plot_three_derivs(raw_deriv, metadata, show_figs, save_figs, figsize=figsize)
    
    

if __name__ == "__main__":
    main()

# TODO: refine Dict[] type hints where there are multiple return types? consider making an object class hmmmm 