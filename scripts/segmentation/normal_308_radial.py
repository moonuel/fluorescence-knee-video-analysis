import sys
import os
import numpy as np
import argparse

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipelines.base import KneeSegmentationPipeline
from src.utils import utils

class Pipeline308(KneeSegmentationPipeline):
    def __init__(self, input_path, preview=True):
        super().__init__(input_path, preview=preview)
        
        # Override Configurations
        self.pre_cfg.crop_size = 500
        self.pre_cfg.rot90_k = -1
        
        self.mask_cfg.otsu_scale = 0.4 
        self.mask_cfg.adaptive_block = 141
        self.mask_cfg.adaptive_c = 14
        self.mask_cfg.morph_erode_kernel = (25, 25)
        self.mask_cfg.morph_open_kernel = (11, 11)
        self.mask_cfg.morph_close_kernel = (27, 27)
        
        self.radial_cfg.n_lines = 128
        self.radial_cfg.n_segments = 64
        self.radial_cfg.tip_range = (0.05, 0.5) 
        self.radial_cfg.midpoint_range = (0.5, 0.95)

    def preprocess(self, video):
        print("Preprocessing video (308 specific)...")
        # 1. Rot90
        if self.pre_cfg.rot90_k:
            video = np.rot90(video, k=self.pre_cfg.rot90_k, axes=(1, 2))
            
        # 2. First crop
        video = utils.center_crop(video, int(500 * np.sqrt(2)))
        
        # 3. Rotate
        video = utils.rotate_video(video, -15)
        
        # 4. Fill empty
        video[video == 0] = 22
        
        # 5. Second crop (rectangular)
        video = utils.center_crop(video, 500, 450)
        
        # 6. Flip
        video = np.flip(video, axis=2)
        
        return video

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

    # Determine input path
    # Assuming running from project root
    input_file = os.path.join("data", "processed", "normal_knee_processed.npy")
    
    if not os.path.exists(input_file):
        print(f"Warning: Input file {input_file} not found.")
        pass

    pipeline = Pipeline308(input_file, preview=not args.no_preview)
    pipeline.run("308") # Pass video_id for saving filename
