"""
Radial segmentation-based analysis of the 1339 aging knee video data.
Segments the knee into N equally-spaced radial slices, and then groups them into chunks associated with the left/middle/right portions of the knee.
"""


import utils.io as io
import utils.views as views
import utils.utils as utils
import core.radial_segmentation as rdl
import core.knee_segmentation as ks
import numpy as np
import cv2

def save_1339_data():
    """Imports, centers, crops, and saves frames 0-650 of the 1339 aging video data. Only needs to be used once"""
    video = io.load_hdf5_video_chunk("../data/raw/right-0 min-regional movement_00001339_grayscale.h5", (0,650), verbose=True)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)

    video, _ = ks.centre_video(video)
    video = utils.crop_video_square(video, 500)

    io.save_nparray(video, "../data/processed/1339_knee_frames_0-649_ctrd.npy")
    return

def load_1339_data() -> np.ndarray:
    """Loads the saved data. See save_1339_data()"""
    return io.load_nparray("../data/processed/1339_knee_frames_0-649_ctrd.npy")

def get_mask_around_femur(video:np.ndarray) -> np.ndarray:
    """Takes the centered grayscale 1339 video and returns a binary mask appropriate for estimating the position of the femur."""

    video = utils.log_transform_video(video)

    blurred = utils.blur_video(video)


    femur_mask = ks.mask_adaptive(blurred, 151, 8)

    otsu_mask = ks.get_otsu_masks(blurred, 0.6)

    # otsu_mask = utils.blur_video(otsu_mask, (35,35))
    # otsu_mask = (otsu_mask > 0).astype(np.uint8) * 255 # clip to binary
    
    # otsu_mask = utils.morph_close(otsu_mask, (25,25))
    # otsu_mask = utils.morph_erode(otsu_mask, (51,51))
    
    # views.show_frames(otsu_mask, "debugging otsu mask params")
    views.draw_mask_boundary(blurred, otsu_mask)

    femur_mask = rdl.interior_mask(otsu_mask, femur_mask)

    # femur_mask = utils.blur_video(femur_mask, (11,11))
    # femur_mask = (femur_mask > 0).astype(np.uint8) * 255 # clip blurred mask to binary mask

    return femur_mask

def main():
    print("main() called!")

    video = load_1339_data()[289:608] # aka 210 - 609, when written in 1-based indexing
    nfs, h, w = video.shape
    
    # agl = 25
    # video = utils.rotate_video(video, agl)

    v_out = views.rescale_video(video, 1, False)

    mask = get_mask_around_femur(video)

    views.show_frames([video, mask])
    views.draw_mask_boundary(video, mask)

    return


if __name__ == "__main__":
    main()