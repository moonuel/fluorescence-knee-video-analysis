from utils import utils, io, views
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
import numpy as np
from functools import partial
import cv2

"""
Steps:
    >>> Get femur mask
    >>> Estimate femur tip
    >>> Estimate femur midpoint
    >>> Get radial segmentation
"""

def main():
    video = io.load_nparray("../data/processed/1193_knee_frames_ctrd.npy")
    views.show_frames(video)

if __name__ == "__main__":
    main()