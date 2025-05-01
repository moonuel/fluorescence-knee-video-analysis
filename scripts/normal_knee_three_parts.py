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
import aging_knee_three_parts as aktp

def load_avi(fn) -> np.ndarray:
    


    return

def main():
    
    # Load video
    video = load_avi("../data/video_1.avi")



if __name__ == "__main__":
    main()