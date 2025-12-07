import os
from pathlib import Path
import numpy as np
from utils import utils, io, views
from pipelines.base import KneeSegmentationPipeline
from core import radial_segmentation as rdl


class Aging1358Segmentation(KneeSegmentationPipeline):

    def preprocess(self, video=None, rot90_k=1, crop_size=500, empty_fill_value=None, inplace=False):
        video = (self.video * 1.95).astype(np.uint8) # brightness adjustment 
        return super().preprocess(video, rot90_k, crop_size, empty_fill_value, inplace)

    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), thresh_scale=0.7, hist_frame=1425, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=(3, 3), morph_close_kernel=(39, 39), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=141, adaptive_c=10, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def refine_femur_mask(self, femur_mask=None, morph_open_kernel=None, morph_close_kernel=(19,19), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        femur_mask = self.femur_mask
        femur_mask[:, :159, :] = 0
        femur_mask[:, 322:, :] = 0
        femur_mask[:, :, :205] = 0
        return super().refine_femur_mask(femur_mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.5), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        """
        Radial segmentation with manual refinements for boundary sampling.
        """

        if mask is None:
            mask = self.otsu_mask
        if femur_mask is None:
            femur_mask = self.femur_mask

        # Apply manual refinements to femur_mask copy for boundary sampling
        boundary_mask = femur_mask.copy()
        boundary_mask[:, :182, :] = 0
        boundary_mask[:, 320:, :] = 0
        boundary_mask[:, :, 308:] = 0

        boundary_points = rdl.sample_femur_interior_pts(boundary_mask, N_lns=n_lines)
        boundary_points = rdl.forward_fill_jagged(boundary_points)

        # Tip
        s, e = tip_range
        tip_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        tip_bndry = rdl.forward_fill_jagged(tip_bndry)
        femur_tip = rdl.get_centroid_pts(tip_bndry)
        femur_tip = rdl.smooth_points(femur_tip, smooth_window)

        # Midpoint
        s, e = midpoint_range
        midpt_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        midpt_bndry = rdl.forward_fill_jagged(midpt_bndry)
        femur_midpt = rdl.get_centroid_pts(midpt_bndry)
        femur_midpt = rdl.smooth_points(femur_midpt, smooth_window)

        radial_labels = rdl.label_radial_masks(mask, femur_tip, femur_midpt, N=n_segments)

        if inplace:
            self.radial_mask = radial_labels

        return radial_labels


if __name__ == "__main__":
    pipeline = Aging1358Segmentation("data/raw/frontal right_00001358.npy", "1358", "aging", 64)
    pipeline.run(debug=True)
