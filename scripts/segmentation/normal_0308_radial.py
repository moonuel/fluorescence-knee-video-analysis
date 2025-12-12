from pipelines.base import KneeSegmentationPipeline
import numpy as np

class normal308(KneeSegmentationPipeline):

    def preprocess(self, video=None, rot90_k=1, rot_angle=-7, crop_size=500, empty_fill_value=22, inplace=False):
        video = self.video
        video = np.flip(video, axis=2)        
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)
    
    def generate_otsu_mask(self, video=None, blur_kernel=(25,25), thresh_scale=0.5, hist_frame=0, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=171, adaptive_c=10, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)
    
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.45), midpoint_range=(0.5, 0.8), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(3,3), morph_close_kernel=(19,19), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        mask = self.femur_mask
        mask[:, 333:, :] = 0
        mask[:, :100, :] = 0
        mask[:, :, :143] = 0
        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

if __name__ == "__main__":
    pipeline = normal308("data/raw/medial fluid movement_00000308.npy", "0308", "normal")
    pipeline.run(debug=True) # Pass video_id for saving filename
