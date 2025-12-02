"""
Radial segmentation-based analysis of the 1339 aging knee video data.
Segments the knee into N equally-spaced radial slices, 
and then groups them into chunks associated with the left/middle/right portions of the knee.

Refinements made in new version:
    1.1 - Adjust the angle of the knee based on the tip boundary.

Comments:
    Results are okay. Maybe these tools will be useful in the future - for now will stick to 1.0 method.
"""


import utils.io as io
import utils.views as views
import utils.utils as utils
import core.data_processing as dp
import core.radial_segmentation as rdl
import core.knee_segmentation as ks
import numpy as np
import cv2
import numpy as np
from skimage.exposure import match_histograms
from config import VERBOSE
from typing import Tuple
import matplotlib.pyplot as plt

def save_1339_data():
    """Imports, centers, crops, and saves frames 0-650 of the 1339 aging video data. Only needs to be used once"""
    video = io.load_hdf5_video_chunk("../data/raw/right-0 min-regional movement_00001339_grayscale.h5", (0,650), verbose=True)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)

    video, _ = ks.centre_video(video)
    video = utils.center_crop(video, 500)

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

    # Get outer mask
    otsu_mask = ks.get_otsu_masks(video_blrd_hist, 0.6)
    otsu_mask = utils.morph_erode(otsu_mask, (41,41))
    
    # views.show_frames(otsu_mask, "debugging otsu mask params")
    # views.draw_mask_boundary(video_blrd_hist, otsu_mask)

    # Get inner mask
    femur_mask = ks.mask_adaptive(video_blrd, 151, 8)

    # Get mask for femur estimation
    femur_mask = rdl.interior_mask(otsu_mask, femur_mask)

    # Refinements
    femur_mask = utils.blur_video(femur_mask, (5,5)) 
    femur_mask = (femur_mask > 0).astype(np.uint8) * 255 # clip blurred mask to binary mask

    return femur_mask

def get_top_boundary_points(sample_pts:np.ndarray) -> np.ndarray:
    """Filters for only the top portion of points on the femur boundary"""
    if VERBOSE: print("get_top_boundary_points() called!")

    sample_pts = np.asarray(sample_pts) # Expected shape: (nfs, npts*, 2), where * indicates jagged dimension
    nfs = sample_pts.shape[0]

    top_pts = []
    for cf in range(nfs):
        cpts = np.asarray(sample_pts[cf])
        top_pts.append(cpts[::2]) # i.e. return every other point 

    return np.array(top_pts, dtype=object)

def get_interior_points(points:np.ndarray, start:float, end:float) -> np.ndarray:
    """Gets the points between the percentiles denoted by start and end"""
    if VERBOSE: print("get_interior_points() called!")

    points = np.asarray(points)
    nfs = points.shape[0] # Expected shape: (nfs, npts*, 2), where * indicates jagged dimension

    intr_pts = []
    for cf in range(nfs):
        cpts = np.asarray(points[cf])
        N = cpts.shape[0]

        ns = int(N*start)
        ne = int(N*end)
        
        intr_pts.append(cpts[ns:ne])

    return np.array(intr_pts, dtype=object)

def fit_line_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a line using PCA (total least squares) to 2D points.

    Parameters:
        points: ndarray of shape (n_points, 2)

    Returns:
        mean: the centroid of the points
        direction: unit vector in the direction of the best-fit line

    """
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]  # First principal component
    return mean, direction

def fit_lines_pca(sample_pts:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a PCA line through a set of points, for every frame

    Args:
        sample_pts (np.ndarray): array of shape (nfs, npts*, 2), where * indicates jagged dimension

    Returns:
        means (np.ndarray): array contaning the mean position of points for every frame
        directions (np.ndarray): array containing unit vector in the direction of best-fit line
    """

    sample_pts = np.asarray(sample_pts)
    nfs = sample_pts.shape[0]

    means = []
    directions = []
    for cf in range(nfs):
        pts = sample_pts[cf]
        mu, dir = fit_line_pca(pts)

        means.append(mu)
        directions.append(dir)

    means = np.array(means)
    directions = np.array(directions)

    return means, directions


def project_point_to_line(q, mean, direction):
    t = np.dot(q - mean, direction)
    return mean + t * direction

def project_points_to_line(points:np.ndarray, means:np.ndarray, directions:np.ndarray) -> np.ndarray:
    """Projects a set of points to the PCA lines specified by means and directions, for every frame 
    
    Args:
        points (np.ndarray): points to be projected to the PCA line
        means (np.ndarray): means at the center of the PCA line
        directions (np.ndarray): unit vectors specifying the orientation of the PCA line

    Returns:
        proj_pts (np.ndarray): points projected onto the PCA line 

    """
    if VERBOSE: print("project_points_to_line() called!")

    points = np.asarray(points)
    points = np.reshape(points, (-1, 2)) # for easier indexing 

    means = np.asarray(means)
    directions = np.asarray(directions)

    assert points.shape == means.shape == directions.shape

    nfs = points.shape[0]

    proj_pts = []
    for cf in range(nfs):
        pt = points[cf]
        mu = means[cf]
        dir = directions[cf]

        prj_pt = project_point_to_line(pt, mu, dir)

        proj_pts.append(prj_pt)

    proj_pts = np.round(proj_pts, 0)
    proj_pts = np.array(proj_pts, dtype=int)
    proj_pts = np.reshape(proj_pts, (-1, 1, 2))

    return proj_pts


def main():
    print("main() called!")

    video = load_1339_data()[289:608] # aka 290 - 609, when written in 1-based indexing
    video = utils.blur_video(video, kernel_dims=(11,11), sigma=3)
    nfs, h, w = video.shape
    
    agl = 22
    video = utils.rotate_video(video, agl)

    # Estimate femur position
    mask = get_mask_around_femur(video)
    femur_bndry = rdl.sample_femur_interior_pts(mask, 128)
    # views.draw_points(video, femur_bndry)

    # Estimate femur tip
    femur_tip_bndry = rdl.estimate_femur_tip_boundary(femur_bndry)
    femur_tip_bndry = rdl.filter_outlier_points_centroid(femur_tip_bndry, eps=60)
    # v1 = views.draw_points(video, femur_tip_bndry)

    femur_tip = rdl.get_centroid_pts(femur_tip_bndry)
    femur_tip = rdl.smooth_points(femur_tip, 10)
    # views.draw_points(v1, femur_tip)

    # Estimate femur midpoint
    femur_mid_bndry = rdl.estimate_femur_midpoint_boundary(femur_bndry, 0.1, 0.4)
    # v2 = views.draw_points(video, femur_mid_bndry, False)
    
    femur_mid = rdl.get_centroid_pts(femur_mid_bndry)
    femur_mid = rdl.smooth_points(femur_mid, 10)    
    # views.draw_points(v2, femur_mid)

    # Midpoint refinement
    femur_top_bndry = get_top_boundary_points(femur_bndry)
    femur_top_bndry = get_interior_points(femur_top_bndry, 0.1, 0.9)

    means, directions = fit_lines_pca(femur_top_bndry)
    
    proj_pts = project_points_to_line(femur_mid, means, directions)
    proj_pts = rdl.smooth_points(proj_pts, 10)

    refined_pts = rdl.get_centroid_pts(np.concatenate([femur_mid, proj_pts], axis=1)) # get midpoint of original and projected points
    print(refined_pts)
    print(refined_pts.shape)

    v3 = views.draw_points(video, proj_pts, False)
    v3 = views.draw_points(v3, femur_mid, False)
    v3 = views.draw_points(v3, refined_pts)
    # femur_mid_ref = 
    # views.draw_points(video, femur_top_bndry)
    
    # return # Temporarily halt execution here 
    
    # Radially segment video
    # circle_pts = rdl.get_N_points_on_circle(femur_tip, femur_mid, N=16, radius_scale=2)
    circle_pts = rdl.get_N_points_on_circle(femur_tip, refined_pts, N=16, radius_scale=2)
    # views.draw_points(v2, circle_pts)

    radial_regions, radial_masks = rdl.get_radial_segments(video, femur_tip, circle_pts, thresh_scale=0.6)
    v1 = views.draw_radial_masks(v3, radial_masks, False)
    views.draw_radial_slice_numbers(v1, circle_pts)

    # return

    # Save segmentation data
    io.save_nparray(video, "../data/processed/1339_knee_radial_video_N16.npy")
    io.save_nparray(radial_masks, "../data/processed/1339_knee_radial_masks_N16.npy")
    io.save_nparray(radial_regions, "../data/processed/1339_knee_radial_regions_N16.npy")

    return


if __name__ == "__main__":
    main()