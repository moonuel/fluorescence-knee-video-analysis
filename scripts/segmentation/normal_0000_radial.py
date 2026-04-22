from pipelines.base import KneeSegmentationPipeline
from core import radial_segmentation as rdl
from utils import utils

class Normal0000Pipeline(KneeSegmentationPipeline):
    
    def preprocess(self, video=None, rot90_k=1, rot_angle=-10, crop_size=500, empty_fill_value=17, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)
    
    def generate_otsu_mask(self, video=None, blur_kernel=(25,25), thresh_scale=0.5, hist_frame=480, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=(21,21), morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=161, adaptive_c=11.5, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        return super().generate_femur_mask(outer_mask, interior_mask, inplace)
    
    def refine_femur_mask(self, mask=None, morph_open_kernel=(5,5), morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        
        mask = self.femur_mask

        mask[:,335:,:] = 0
        mask[:,:161,:] = 0
        mask[:,:,:174] = 0
        mask[:, 294:335, 174:343] = 255 # y = 294
        mask[:, 252:294, 212:249] = 0 # box from 212,270 to 249,294
        
        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)
    
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, 
                            tip_range=(0.05, 0.4), midpoint_range=(0.7, 0.95), smooth_window=9, *, 
                            tip_x_weight = 0.4, tip_y_weight = 0.45, midpoint_x_weight = 0.4, midpoint_y_weight = 0.5, centroid_mode = "quantile", inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, tip_x_weight=tip_x_weight, tip_y_weight=tip_y_weight, midpoint_x_weight=midpoint_x_weight, midpoint_y_weight=midpoint_y_weight, centroid_mode=centroid_mode, inplace=inplace)

    def run(self, debug=False, debug_pause=False):
        return super().run(debug, debug_pause)
    
if __name__ == "__main__":
    pipeline = Normal0000Pipeline(r"data\raw\right before_00000000.npy", "0000", "normal")
    pipeline.run(debug=True, debug_pause=False)