import os
from pathlib import Path
import numpy as np
from utils import utils, io, views
from pipelines.base import KneeSegmentationPipeline
from core import radial_segmentation as rdl


class Aging1358Segmentation(KneeSegmentationPipeline):

    def preprocess(self, video=None, rot90_k=1, rot_angle=None, crop_size=500, empty_fill_value=None, inplace=False):
        # video = (self.video * 1.95).astype(np.uint8) # brightness adjustment
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)

    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), thresh_scale=0.7, hist_frame=1425, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=(3, 3), morph_close_kernel=(15,15), morph_erode_kernel=None, morph_dilate_kernel=(11,11), inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=141, adaptive_c=6, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def refine_femur_mask(self, femur_mask=None, morph_open_kernel=(3,3), morph_close_kernel=(19,19), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        femur_mask = self.femur_mask
        femur_mask[:, :159, :] = 0
        femur_mask[:, 322:, :] = 0
        femur_mask[:, :, :205] = 0
        return super().refine_femur_mask(femur_mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.4), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

if __name__ == "__main__":
    pipeline = Aging1358Segmentation("data/raw/frontal right_00001358.npy", "1358", "aging", 64)
    pipeline.run(debug=True)
