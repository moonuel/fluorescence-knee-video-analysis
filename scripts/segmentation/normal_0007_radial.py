from pipelines.base import KneeSegmentationPipeline
from core import radial_segmentation as rdl
from utils import utils
import numpy as np

class Normal0007Pipeline(KneeSegmentationPipeline):
    
    def preprocess(self, video=None, rot90_k=1, rot_angle=-24, crop_size=500, empty_fill_value=17, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)
    
    def generate_otsu_mask(self, video=None, blur_kernel=(35,35), thresh_scale=0.5, hist_frame=956, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=(21,21), morph_dilate_kernel=None, inplace=False):
        self.video_hist = np.clip(self.video_hist.astype(np.int16) - 17, 0, 255).astype(np.uint8)
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=163, adaptive_c=12, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        return super().generate_femur_mask(outer_mask, interior_mask, inplace)
    
    def refine_femur_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=(3,3), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        mask = self.femur_mask

        # Include only knee region
        mask[:,347:,:] = 0
        mask[:,:180,:] = 0
        mask[:,:,:185] = 0
        # mask[:, :, 340:] = 0

        # Fill mask gaps with boxes
        mask[:, 313:347, 184:366] = 255 # MCL from (184,313) to (366,347)
        mask[:, 247:346, 340:365] = 255 # Lower-right joint cavity from (340, 247) to (365,265) 

        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)
    
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, 
                            tip_range=(0.05, 0.45), midpoint_range=(0.50, 0.95), smooth_window=11, *, 
                            tip_x_weight = 0.50, tip_y_weight = 0.50, midpoint_x_weight = 0.5, midpoint_y_weight = 0.50, 
                            centroid_mode = "quantile", inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, tip_x_weight=tip_x_weight, tip_y_weight=tip_y_weight, midpoint_x_weight=midpoint_x_weight, midpoint_y_weight=midpoint_y_weight, centroid_mode=centroid_mode, inplace=inplace)

    def run(self, debug=False, debug_pause=False):
        return super().run(debug, debug_pause)
    
if __name__ == "__main__":
    print("Warning: Normal0007 segmentation is not recommended for use.")
    pipeline = Normal0007Pipeline(r"data\raw\right before_00000007.npy", "0007", "normal")
    pipeline.run(debug=True, debug_pause=False)