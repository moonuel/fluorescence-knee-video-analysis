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
    video = utils.crop_video_square(video, 350)
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.blur_video(video, (31,31), 0)
    # video = cv2.GaussianBlur(video, (31,31), 0)

    views.view_frames(video)


if __name__ == "__main__":
    main()