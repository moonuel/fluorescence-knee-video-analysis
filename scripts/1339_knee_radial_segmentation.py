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
import numpy as np
from skimage.exposure import match_histograms

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

def match_histograms_video(video, reference_frame=None):
    """
    Apply histogram matching to each frame in the video.
    
    Args:
        video: np.ndarray of shape (n_frames, height, width)
        reference_frame: optional np.ndarray (height, width), default is video[0]

    Returns:
        matched_video: np.ndarray of same shape as input
    """
    if reference_frame is None:
        reference_frame = video[0]
        
    matched_video = np.empty_like(video)
    for i in range(video.shape[0]):
        matched_video[i] = match_histograms(video[i], reference_frame)
        
    return matched_video


def get_mask_around_femur(video:np.ndarray) -> np.ndarray:
    """Takes the centered grayscale 1339 video and returns a binary mask appropriate for estimating the position of the femur."""

    video_blrd = utils.blur_video(video)
    video_blrd_hist = match_histograms_video(video_blrd) # For consistency of Otsu segmentation

    otsu_mask = ks.get_otsu_masks(video_blrd_hist, 0.6)
    otsu_mask = utils.morph_erode(otsu_mask, (41,41))
    
    # views.show_frames(otsu_mask, "debugging otsu mask params")
    views.draw_mask_boundary(video_blrd_hist, otsu_mask)

    femur_mask = ks.mask_adaptive(video_blrd, 151, 8)

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