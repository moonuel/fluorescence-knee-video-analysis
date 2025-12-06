from pipelines.base import KneeSegmentationPipeline
from utils import io, utils, views
import os

class Segmentation1357(KneeSegmentationPipeline):

    # Override parent methods using optimized parameters
    def preprocess(self, video=None, rot90_k=1, crop_size=500, empty_fill_value=16, inplace=False):
        return super().preprocess(video, rot90_k, crop_size, empty_fill_value, inplace)

    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), thresh_scale=0.7, hist_frame=67, inplace=False):
        return super().generate_otsu_mask(video=video, blur_kernel=blur_kernel, thresh_scale=thresh_scale, hist_frame=hist_frame, inplace=inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_erode_kernel, morph_dilate_kernel, morph_open_kernel, morph_close_kernel, inplace)

    def generate_interior_mask(self, hist_video=None, adaptive_block=141, adaptive_c=5.3, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block, adaptive_c, inplace)

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        return super().generate_femur_mask(outer_mask=outer_mask, interior_mask=interior_mask, inplace=inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(5,5), morph_close_kernel=(15,15), morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        # Manual cuts 
        mask = self.femur_mask
        mask[:, 305:, :] = 0
        mask[:, :177, :] = 0
        mask[:, :, :204] = 0
        return super().refine_femur_mask(mask, morph_erode_kernel, morph_dilate_kernel, morph_open_kernel, morph_close_kernel, inplace)

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.5), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        return super().radial_segmentation(mask=mask, femur_mask=femur_mask, n_lines=n_lines, n_segments=n_segments,
                                         tip_range=tip_range, midpoint_range=midpoint_range, smooth_window=smooth_window, inplace=inplace)


def main():
    """
    Demonstrates the new debug workflow:
    1. Metadata (video_id, condition, n_segments) specified at initialization
    2. Override methods with custom parameters in subclass
    3. Use run(debug=True) to visualize all intermediate steps
    4. Same code works for both debugging and production
    """
    input_path = os.path.join("data", "raw", "right_00001357.npy")

    # Initialize with required metadata - no longer passed to run()
    aging1357 = Segmentation1357(
        input_path=input_path,
        video_id="1357",
        condition="aging",
        n_segments=64
    )

    # Debug mode: visualize all intermediate steps using overridden methods
    print("Running pipeline in debug mode with optimized parameters...\n")
    aging1357.run(debug=True, debug_pause=False)

    # For step-by-step debugging with pauses:
    # aging1357.run(debug=True, debug_pause=True)

    # For production (clean execution, no intermediate visualization):
    # aging1357.run()

    return

if __name__ == "__main__":
    main()
