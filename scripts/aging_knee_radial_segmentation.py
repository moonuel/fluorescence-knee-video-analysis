import os
import sys
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import src.core.knee_segmentation as ks
from src.utils import io, views, utils
from src.config import VERBOSE

def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, _ = knee.pre_process_video(video) 
    video = io.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call

    # Pre-process video
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.crop_video_square(video, int(350*np.sqrt(2))) 

    # Slight rotation
    angle = -36
    video = utils.rotate_video(video, angle)
    video = utils.crop_video_square(video, 350) # crop out black

    # Get adaptive mean mask
    video_blr = utils.blur_video(video, (31,31), 0)
    mask = utils.mask_adaptive(video_blr, 71, -2)
    mask = utils.morph_open(mask, (15,15)) # clean small artifacts

    views.draw_middle_lines(mask, show_video=True)
    views.draw_middle_lines(video, show_video=True)

    views.view_frames(video)

if __name__ == "__main__":
    main()