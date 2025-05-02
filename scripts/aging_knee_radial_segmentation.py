import os
import sys
import numpy as np
import pandas as pd
import cv2
import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import src.core.knee_segmentation as ks

VERBOSE=True

def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, translation_mxs = knee.pre_process_video(video) 
    video = ks.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call
    translation_mxs = ks.load_nparray("../data/processed/translation_mxs.npy")

    ks.view_frames(video)

if __name__ == "__main__":
    main()