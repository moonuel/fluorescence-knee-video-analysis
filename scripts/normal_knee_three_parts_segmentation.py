import os
import sys
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import src.core.knee_segmentation as ks
from src.utils import io, utils, views
from src.core import data_processing as dp
from src.config import VERBOSE

def main():
    if VERBOSE: print("main() called!")

    # Load and preprocess video
    # video = load_avi("../data/video_1.avi")
    # video, translation_mxs = ks.centre_video(video)
    video = io.load_nparray("../data/processed/normal_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/normal_translation_mxs.npy")

    # Load and transform coords
    coords, metadata = io.load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", sheet_num=3)
    coords = ks.translate_coords(translation_mxs, coords)
    # coords = ks.smooth_coords(coords, 5) # Smooth coords

    # Segment video
    regions, masks = ks.get_three_segments(video, coords, thresh_scale=0.65)

    # Plot intensities
    keys=['l','m','r']
    show_figs=True
    save_figs=True
    figsize=(10,15)

    raw_intensities = dp.measure_region_intensities(regions, masks, keys)
    normalized_intensities = dp.measure_region_intensities(regions, masks, keys, normalized=True)

    views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)
    views.plot_three_intensities(normalized_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)

    # Plot rates of change
    raw_deriv = dp.get_intensity_diffs(raw_intensities)
    views.plot_three_derivs(raw_deriv, metadata, show_figs, save_figs, figsize)



if __name__ == "__main__":
    main()

# TODO: wrap coords data in a Coords object to enforce the existence of certain metadata? 
# ^ could subclass the Coords object to have more descriptive names based on the sheet that is selected? idk 