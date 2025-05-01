import os
import sys
module_path = os.path.abspath(os.path.join('..', 'utils')) # Build an absolute path from this notebook's parent directory
if module_path not in sys.path: # Add to sys.path if not already present
    sys.path.append(module_path)
import numpy as np
import pandas as pd
import cv2
import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import aging_knee_three_parts as aktp

VERBOSE = True

def load_avi(fn) -> np.ndarray:
    if VERBOSE: print("load_avi() called!")

    cap = cv2.VideoCapture(fn)
    video = []
    while cap.isOpened():

        ret, frame = cap.read() # Returns (boolean, bgr frame)
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        video.append(frame)

    cap.release()
    video = np.array(video)
    return video

def load_normal_knee_coords(fn:str, sheet_num:int) -> pd.DataFrame:
    if VERBOSE: print("load_normal_knee_coords() called!")

    sheet_names=["8.29 re-measure", "8.29 2nd", "8.29 3rd", "8.6"]
    coords = pd.read_excel(fn, engine="openpyxl", sheet_name=sheet_num, usecols="B,D,E")

    coords["Frame Number"].ffill(inplace=True)    
    coords.set_index("Frame Number", inplace=True)

    assert coords.isnull().values.any() == 0

    flx_ext_pts = [117, 299, 630, 117]
    uqf = coords.index.unique()

    metadata = {"knee_name": sheet_names[sheet_num], "flx_ext_pt": flx_ext_pts[sheet_num], "f0": uqf[0], "fN": uqf[-1]}

    return coords, metadata

def main():
    if VERBOSE: print("main() called!")

    # Load video
    video = load_avi("../data/video_1.avi")

    # Load coords
    coords, metadata = load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", 3)

    # Preprocess video
    video, translation_mxs = aktp.pre_process_video(video)

    # Transform coords
    coords = aktp.translate_coords(translation_mxs, coords)

    # Segment video
    regions, masks = aktp.get_three_segments(video, coords)

    # Plot intensities
    raw_intensities = aktp.measure_region_intensities(regions, masks, ['l','m','r'])
    aktp.plot_three_intensities(raw_intensities, metadata)



if __name__ == "__main__":
    main()

# TODO: wrap coords data in a Coords object to enforce the existence of certain metadata? 
# ^ could subclass the Coords object to have more descriptive names based on the sheet that is selected? idk 