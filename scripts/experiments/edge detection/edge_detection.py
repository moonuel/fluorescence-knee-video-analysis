from utils import io, views, utils
import cv2
import pandas as pd
import numpy as np
import scipy.ndimage as spi


# Select arbitrary frame for testing
video = io.load_nparray("../../data/segmented/normal_0308_radial_video_N64.npy")
video = utils.blur_video(video, (25,25), 5)
frame = video[100]

# Compute discrete gradient 
Kx = np.array([[1, 0, -1], # Sobel kernels 
               [2, 0, -2],
               [1, 0, -1]])
Ky = Kx.T

Gx = spi.convolve(frame.astype(int), Kx) # gradients 
Gy = spi.convolve(frame.astype(int), Ky)

G = np.hypot(Gx, Gy) # magnitude

theta = np.arctan2(Gy, Gx) # direction

views.show_frames([frame, G, Gx-np.min(Gx), Gy-np.min(Gy)], r"308 frame 100, |G|, Gx, Gy")

breakpoint()
