import sys
import os
import numpy as np
import argparse

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipelines.base import KneeSegmentationPipeline
from src.utils import utils

class normal308(KneeSegmentationPipeline):

    def preprocess(self, video=None, rot90_k=-1, rot_angle=-15, crop_size=500, empty_fill_value=22, inplace=False):
        print("Preprocessing video (308 specific)...")
        video = self.video
        video = np.flip(video, axis=2)
        return video
    
    def generate_otsu_mask(self, video=None, blur_kernel=(25,25), thresh_scale=0.5, hist_frame=0, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=141, adaptive_c=14, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)
    
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.45), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

    def refine_femur_mask(self, mask):
        print("Applying manual cuts for 308...")
        mask[:, 330:, :] = 0
        mask[:, :121, :] = 0
        mask[:, :, :143] = 0
        return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline for video 308")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview and confirmation")
    args = parser.parse_args()

    input_file = os.path.join("data", "processed", "normal_knee_processed.npy")
 

    pipeline = normal308("data/raw/medial fluid movement_00000308.npy", "0308", "normal")
    pipeline.run(debug=True) # Pass video_id for saving filename
