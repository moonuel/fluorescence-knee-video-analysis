import os
import sys
import numpy as np
import pandas as pd
import cv2
from src.utils import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import src.core.knee_segmentation as ks
from src.utils import io, views
from src.config import VERBOSE


def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, translation_mxs = knee.pre_process_video(video) 
    video = io.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call
    translation_mxs = io.load_nparray("../data/processed/translation_mxs.npy")

    # Process coords
    coords = io.load_aging_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", 2)

    views.view_frames(video)

if __name__ == "__main__":
    main()