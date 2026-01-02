from pipelines.base import KneeSegmentationPipeline
from core import radial_segmentation as rdl
from utils import utils


class Normal1190(KneeSegmentationPipeline):
    """Segmentation pipeline for normal knee 1190 using the new interface.

    Mapped from legacy normal_1190_radial.py:
    - Preprocessing: center crop 500, rot90=1, fill empty pixels with 18
      (no extra rotation).
    - Otsu outer mask: histogram match to frame 492, thresh_scale=0.6,
      then a relatively strong erode (41x41).
    - Interior mask: adaptive block=141, C=6.5.
    - Femur mask refinement: open(11,11), close(27,23) plus manual cuts in
      y and x matching the legacy cropping logic.
    - Radial segmentation: use base implementation with parameters tuned to
      match legacy choices: n_lines=128, n_segments=64, tip range ~0.6,
      midpoint range (0.1, 0.4), smoothing window=15.
    """

    # ------------------------
    # Preprocessing
    # ------------------------
    def preprocess(self, video=None, rot90_k=1, rot_angle=None, crop_size=500,
                   empty_fill_value=18, inplace=False):
        return super().preprocess(video, rot90_k, rot_angle, crop_size,
                                  empty_fill_value, inplace)

    # ------------------------
    # Otsu / Outer Mask
    # ------------------------
    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25),
                           thresh_scale=0.7, hist_frame=492, inplace=False):
        return super().generate_otsu_mask(video, blur_kernel,
                                          thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=None,
                         morph_close_kernel=(31,31), morph_erode_kernel=None,
                         morph_dilate_kernel=None, inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel,
                                        morph_close_kernel, morph_erode_kernel,
                                        morph_dilate_kernel, inplace)

    # ------------------------
    # Interior / Femur Mask
    # ------------------------
    def generate_interior_mask(self, hist_video=None, adaptive_block=151,
                               adaptive_c=3, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block,
                                              adaptive_c, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(5,5),
                          morph_close_kernel=(15,15), morph_erode_kernel=None,
                          morph_dilate_kernel=None, inplace=False):
        mask = self.femur_mask
        mask = utils.morph_dilate(mask, (11,11))
        mask[:, 341:, :] = 0
        mask[:, :132, :] = 0
        mask[:, :, :189] = 0
        mask[:, :, 374:] = 0
        # mask[:, :, 291:] = 0
        # mask[445:475, :, 278:] = 0
        # mask[521:540, :, 282:] = 0

        return super().refine_femur_mask(mask, morph_open_kernel,
                                         morph_close_kernel, morph_erode_kernel,
                                         morph_dilate_kernel, inplace)

    # ------------------------
    # Radial Segmentation
    # ------------------------
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128,
                            n_segments=64, tip_range=(0.05, 0.4),
                            midpoint_range=(0.5, 0.95), smooth_window=9,
                            inplace=False):
        return super().radial_segmentation(mask, femur_mask, n_lines,
                                           n_segments, tip_range,
                                           midpoint_range, smooth_window,
                                           inplace)


if __name__ == "__main__":
    pipeline = Normal1190(
        input_path="data/raw/1 con-0 min-fluid movement_00001190.npy",
        video_id="1190",
        condition="normal",
        n_segments=64,
    )
    pipeline.run(debug=True)
