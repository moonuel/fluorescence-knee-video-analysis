import tifffile as tf
import numpy as np
from utils import io, views, utils
import cv2
import core.radial_segmentation as rdl
from pipelines.base import KneeSegmentationPipeline

class DMM_4W(KneeSegmentationPipeline):

    def __init__(self, input_path, video_id, condition, n_segments=64, output_dir=None):
        super().__init__(input_path, video_id, condition, n_segments, output_dir)

    def preprocess(self, video=None, rot90_k=1, crop_size=500, empty_fill_value=None, inplace=False):
        video = self.video
        video *= 3
        video = utils.rotate_video(video,-10)
        return super().preprocess(video, rot90_k, crop_size, empty_fill_value, inplace)

    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), thresh_scale=0.9, hist_frame=17, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=(31,31), inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)
    
    def generate_interior_mask(self, hist_video=None, adaptive_block=161, adaptive_c=12, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)
    
    def refine_femur_mask(self, mask=None, morph_open_kernel=(3,3), morph_close_kernel=(31,31), morph_erode_kernel=(21,21), morph_dilate_kernel=None, inplace=False):
        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.3), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

if __name__ == "__main__":
    pipeline = DMM_4W("data/raw/dmm 4w 550 frames 17 cycles 1_00000091.npy", "4w", "dmm", 64)
    pipeline.run(debug=True)
