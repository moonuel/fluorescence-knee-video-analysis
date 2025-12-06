import os
import sys
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import core.knee_segmentation as ks
from utils import io, utils, views
from core import data_processing as dp
from config import VERBOSE

def main():
    if VERBOSE: print("main() called!")

    # Load and preprocess video
    video = io.load_nparray("../data/segmented/normal_knee_processed.npy") # Centered video
    video = utils.blur_video(video, (25,25), 3) # equivalent to ndimage.gaussian_filter(video, sigma=3)?
    translation_mxs = io.load_nparray("../data/segmented/normal_translation_mxs.npy") # Corresponding translations

    # Load and transform coords
    # TODO: the coordinate data corresponding to sheet_num=2 is not contiguous. there is a discontinuity between frames 618 and 630.
            # How did Juan deal with this? maybe he freezes the last-used coords?
            # I think matplotlib just draws a line between the prev and current values
    # TODO: continue investigating discrepancies in me and Juan's plots
    coords, metadata = io.load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", sheet_num=1) 
    coords = ks.translate_coords(translation_mxs, coords)
    # coords = ks.smooth_coords(coords, 5) # Smooth coords

    # Segment video
    regions, masks = ks.get_three_segments(video, coords, thresh_scale=0.65)
    views.show_regions(regions, ['l','m','r'])

    # Plot intensities
    keys=['l','m','r']
    show_figs=True
    save_figs=False
    figsize=(9,3)

    raw_intensities = dp.measure_region_mean_intensities(regions, masks, keys)

    views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs, vert_layout=False, figsize=figsize)

if __name__ == "__main__":
    main()
