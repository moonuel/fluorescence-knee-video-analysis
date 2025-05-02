import os
import sys
module_path = os.path.abspath(os.path.join('..', 'utils')) # Build an absolute path from this notebook's parent directory
if module_path not in sys.path: # Add to sys.path if not already present
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import cv2
from tifffile import imread as tif_imread
import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import aging_knee_three_parts as knee

VERBOSE=True

def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, translation_mxs = knee.pre_process_video(video) 
    video = knee.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call
    translation_mxs = knee.load_nparray("../data/processed/translation_mxs.npy")

    knee.view_frames(video)

if __name__ == "__main__":
    main()