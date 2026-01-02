from pipelines.base import KneeSegmentationPipeline
from utils import utils


class Normal1193(KneeSegmentationPipeline):
    """Segmentation pipeline for normal knee 1193 using the new interface.

    Mapped from legacy normal_1193_radial.py:
    - Preprocessing: center crop 500, rot90=1, fill empty pixels with 19.
    - Otsu outer mask: histogram match to frame ~118, thresh_scale=0.6,
      followed by strong erosion.
    - Interior mask: adaptive block=141, C=8.
    - Femur mask refinement: blur + binarize + close + manual cuts in t/y/x
      matching the legacy script.
    - Radial segmentation: reuse base implementation with parameters tuned
      to approximate legacy behavior (128 lines, 64 segments, tighter
      smoothing, adjusted tip/midpoint ranges).
    """

    # ------------------------
    # Preprocessing
    # ------------------------
    def preprocess(self, video=None, rot90_k=1, rot_angle=None, crop_size=500,
                   empty_fill_value=19, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size,
                                  empty_fill_value, inplace)

    # ------------------------
    # Otsu / Outer Mask
    # ------------------------
    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25),
                           thresh_scale=0.6, hist_frame=118, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel,
                                          thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=None,
                         morph_close_kernel=None, morph_erode_kernel=(41, 41),
                         morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel,
                                        morph_close_kernel,
                                        morph_erode_kernel,
                                        morph_dilate_kernel, inplace)

    # ------------------------
    # Interior / Femur Mask
    # ------------------------
    def generate_interior_mask(self, hist_video=None, adaptive_block=141,
                               adaptive_c=8, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block,
                                              adaptive_c, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(11,11),
                          morph_close_kernel=(25, 25), morph_erode_kernel=None,
                          morph_dilate_kernel=None, inplace=False):

        mask = self.femur_mask

        # Manual cuts in t/y/x indices
        # Frames 610–789
        mask[610:790, 321:, :] = 0
        mask[610:790, :148, :] = 0
        # mask[610:790, :, 312:] = 0

        # Frames 1700+
        mask[1700:, 333:, :] = 0
        mask[1700:, :127, :] = 0
        # mask[1700:, :, 335:] = 0

        return super().refine_femur_mask(mask,
                                         morph_open_kernel,
                                         morph_close_kernel,
                                         morph_erode_kernel,
                                         morph_dilate_kernel,
                                         inplace)

    # ------------------------
    # Radial Segmentation
    # ------------------------
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128,
                            n_segments=64, tip_range=(0.05, 0.4),
                            midpoint_range=(0.6, 0.95), smooth_window=7,
                            inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines,
                                           n_segments, tip_range,
                                           midpoint_range, smooth_window,
                                           inplace)


if __name__ == "__main__":
    pipeline = Normal1193(
        input_path="data/raw/1 con-10 min slow and quick joint movement tiny movement with quick_00001193.npy",
        video_id="1193",
        condition="normal",
        n_segments=64,
    )
    pipeline.run(debug=True)
