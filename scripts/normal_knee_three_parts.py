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

    coords = pd.read_excel(fn, engine="openpyxl", usecols="B,D,E")

    coords["Frame Number"].ffill(inplace=True)    
    coords.set_index("Frame Number")
    coords.drop("Frame Number", axis=1, inplace=True)

    assert coords.isnull().values.any() == 0

    flx_ext_pts = [117, 299, 630, 117]
    uqf = coords.index.unique()

    metadata = {"knee_name": NotImplemented, "flx_xt_pt": flx_ext_pts[sheet_num], "f0": uqf[0], "fN": uqf[-1]}

    return coords, metadata

def main():
    if VERBOSE: print("main() called!")

    # Load video
    video = load_avi("../data/video_1.avi")

    # Load coords
    coords, metadata = load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", 3)

    




if __name__ == "__main__":
    main()

# TODO: wrap coords data in a Coords object to enforce the existence of certain metadata? 
# ^ could subclass the Coords object to have more descriptive names based on the sheet that is selected? idk 