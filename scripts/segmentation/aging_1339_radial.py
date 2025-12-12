from pipelines.base import KneeSegmentationPipeline
from utils import utils

class Aging1339(KneeSegmentationPipeline):

    def __init__(self, input_path, video_id, condition, n_segments=64, output_dir=None):
        super().__init__(input_path, video_id, condition, n_segments, output_dir)

    def preprocess(self, video=None, rot90_k=1, rot_angle=-26, crop_size=500, empty_fill_value=17, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size, empty_fill_value, inplace)

    def generate_otsu_mask(self, video=None, blur_kernel=(25,25), thresh_scale=0.55, hist_frame=198, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel, thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=(25,25), morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=(21,21), inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=161, adaptive_c=12, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        outer_mask = utils.morph_erode(self.otsu_mask, (75,75))
        return super().generate_femur_mask(outer_mask, interior_mask, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=(35,35), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        femur_mask = self.femur_mask
        femur_mask[:, 352:, :] = 0
        femur_mask[:, :157, :] = 0
        femur_mask[:, :, :205] = 0
        return super().refine_femur_mask(mask, morph_open_kernel, morph_close_kernel, morph_erode_kernel, morph_dilate_kernel, inplace)

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05,0.5), midpoint_range=(0.6,0.9), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines, n_segments, tip_range, midpoint_range, smooth_window, inplace)

if __name__ == "__main__":
    pipeline = Aging1339("data/raw/right-0 min-regional movement_00001339.npy", "1339", "aging")
    pipeline.run(debug=True)