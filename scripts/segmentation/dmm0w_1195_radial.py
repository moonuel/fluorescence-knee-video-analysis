import tifffile as tf
import numpy as np
from utils import io, views, utils
import cv2
import core.radial_segmentation as rdl
from pipelines.base import KneeSegmentationPipeline

class DMM1195(KneeSegmentationPipeline):

    def __init__(self, input_path, video_id, condition, n_segments=64, output_dir=None):
        super().__init__(input_path, video_id, condition, n_segments, output_dir)

    def preprocess(self, video=None, rot90_k=1, rot_angle=-18, crop_size=500, empty_fill_value=20, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)
    
    def generate_otsu_mask(self, video=None, blur_kernel=(25,25), thresh_scale=0.5, hist_frame=155, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=(5,5), morph_close_kernel=(21,21), morph_erode_kernel=(19,19), morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)
    
    def generate_interior_mask(self, hist_video=None, adaptive_block=161, adaptive_c=10, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        outer_mask = self.otsu_mask
        outer_mask = utils.morph_erode(outer_mask, (25,25))
        return super().generate_femur_mask(outer_mask, interior_mask, inplace)
    
    def refine_femur_mask(self, mask=None, morph_open_kernel=(5,5), morph_close_kernel=(19,19), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        mask = self.femur_mask
        mask[:,:140,:] = 0
        mask[:,300:,:] = 0
        mask[:,:,362:] = 0
        mask[:,:,:184] = 0
        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)
    
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.5), midpoint_range=(0.55, 0.95), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

if __name__ == "__main__":
    pipeline = DMM1195("data/raw/1 con-20 min-fluid movement recovery slow and qucik joint motion_00001195.npy", "1195", "dmm0w", 64)
    pipeline.run(debug=True)
    # pipeline.display_saved_results()
