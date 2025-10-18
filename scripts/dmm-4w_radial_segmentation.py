import tifffile as tf
import numpy as np
from utils import io, views, utils
import cv2
import core.radial_segmentation as rdl


def load_dmm_4w(preview:bool=False) -> np.ndarray:

    print("Loading DMM-4W...")
    video = tf.imread(r"E:\Knee Fluid Analysis\New Data\Osteoarthritis model\4w\dmm 4w 550 frames 17 cycles 1_00000091.tif")
    video = np.rot90(video, -1, (2, 1))
    print(f"DMM-4W.shape={video.shape}")

    if preview: views.show_frames(video, "DMM-4W Preview")
    
    return video


def centre_dmm_4w(video:np.ndarray):

    # Adjust contrast
    video = video * 3
    views.show_frames(video, "DMM 4w Contrast Adjusted")

    # Stabilize
    video, _ = rdl.centre_video_mp(video, 10)
    views.show_frames(video, "Stabilized DMM 4w")

    # Crop
    video = utils.crop_video_square(video, 450)
    views.show_frames(video, "DMM 4w Cropped")

    breakpoint()

    return video


def main():

    video = load_dmm_4w(True)
    video_ctrd = centre_dmm_4w(video)



if __name__ == "__main__":
    main()