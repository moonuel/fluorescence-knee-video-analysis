from pipelines.base import KneeSegmentationPipeline
from utils import utils
from core import radial_segmentation as rdl


class Aging1342Segmentation(KneeSegmentationPipeline):
    """Segmentation pipeline for aging knee 1342 using the new interface.

    Parameter mapping notes vs. legacy script:
    - Preprocessing: center crop 500, rot90_k=1, rotate +8 deg, fill empty=16.
      (Matches `load_video()` in old script; the femur-mask-specific horizontal flip
       is *not* used for the final radial masks, so we omit it here.)
    - Otsu mask: blur + histogram match to frame 76, thresh_scale=0.8,
      morphological open (29, 29) as in `get_otsu_mask`.
    - Interior mask: adaptive threshold with block=141, C=8 as in `get_femur_mask`.
    - Femur mask: intersection of outer + interior, then open(15, 15), close(25, 25)
      to approximate the legacy femur mask pipeline.
    - Radial segmentation: manual cuts on femur mask before boundary sampling
      (rows <182, rows >=320, cols >=308 set to 0), N_lns=128, N_segments=64,
      tip range 0.1–0.5 and midpoint range 0.6–1.0 using midpoint-boundary
      estimator for both tip and midpoint (matching legacy behavior).
    """

    # ------------------------
    # Preprocessing
    # ------------------------
    def preprocess(self, video=None, rot90_k=1, rot_angle=-12, crop_size=500,
                   empty_fill_value=16, inplace=False):
        """Apply rotation, crop, and border fill.

        Legacy mapping:
        - utils.center_crop(video, 500)
        - np.rot90(video, k=1, axes=(1, 2))
        - utils.rotate_video(video, 8)
        - video[video == 0] = 16
        """
        return super().preprocess(video, rot90_k, rot_angle, crop_size,
                                  empty_fill_value, inplace)

    # ------------------------
    # Otsu / Outer Mask
    # ------------------------
    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25),
                           thresh_scale=0.8, hist_frame=76, inplace=False):
        """Outer mask using Otsu with histogram matching to frame 76.

        Legacy mapping:
        - video = utils.blur_video(video)
        - video_hist = rdl.match_histograms_video(video, video[76])
        - otsu_mask = ks.get_otsu_masks(video_hist, 0.8)
        - otsu_mask = utils.morph_open(otsu_mask, (29, 29))
        (We split the open step into refine_otsu_mask.)
        """
        return super().generate_otsu_mask(video, blur_kernel,
                                          thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=(29, 29),
                         morph_close_kernel=None, morph_erode_kernel=None,
                         morph_dilate_kernel=(21,21), inplace=False):
        """Apply morphological opening as in legacy get_otsu_mask."""
        return super().refine_otsu_mask(mask, morph_open_kernel,
                                        morph_close_kernel, morph_erode_kernel,
                                        morph_dilate_kernel, inplace)

    # ------------------------
    # Interior / Femur Mask
    # ------------------------
    def generate_interior_mask(self, hist_video=None, adaptive_block=141,
                               adaptive_c=8, inplace=False):
        """Interior mask using adaptive thresholding.

        Legacy mapping from get_femur_mask:
        - inner_mask = ks.mask_adaptive(video_hist, 141, 8)
        """
        return super().generate_interior_mask(hist_video, adaptive_block,
                                              adaptive_c, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(15, 15),
                          morph_close_kernel=(21, 21), morph_erode_kernel=None,
                          morph_dilate_kernel=None, inplace=False):
        """Approximate legacy femur_mask refinement.

        Legacy steps:
        - femur_mask = morph_open(femur_mask, (15, 15))
        - femur_mask = morph_close(femur_mask, (25, 25))
        - femur_mask = blur_video(femur_mask)  # then threshold >127

        Here we keep open+close. If you want, we could also add a mild blur
        before thresholding in a custom override.
        """
        # mask = self.femur_mask
        # mask[:,:193,:] = 0
        # mask[:,327:,:] = 0
        # mask[:,:,:189] = 0
        # mask[:,:,348:] = 0
        return super().refine_femur_mask(mask, morph_open_kernel,
                                         morph_close_kernel, morph_erode_kernel,
                                         morph_dilate_kernel, inplace)

    # ------------------------
    # Radial Segmentation
    # ------------------------
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128,
                            n_segments=64, tip_range=(0.05, 0.5),
                            midpoint_range=(0.5, 0.8), smooth_window=9,
                            inplace=False):
        """Radial segmentation with manual boundary cuts.

        Legacy mapping:
        - Manual femur mask cuts before sampling boundary points:
            mask[:, :182, :] = 0
            mask[:, 320:, :] = 0
            mask[:, :, 308:] = 0
        - boundary_points = rdl.sample_femur_interior_pts(mask, N_lns=128)
        - femur_tip  = estimate_femur_midpoint(boundary_points, 0.1, 0.5)
        - femur_midpt = estimate_femur_midpoint(boundary_points, 0.6, 1.0)
        - radial_masks = rdl.label_radial_masks(otsu_mask, femur_tip, femur_midpt, N=64)
        """
        if mask is None:
            mask = self.otsu_mask
        if femur_mask is None:
            femur_mask = self.femur_mask

        # Apply manual refinements for boundary sampling on a copy
        boundary_mask = femur_mask.copy()
        boundary_mask[:, :182, :] = 0
        boundary_mask[:, 320:, :] = 0
        boundary_mask[:, :, 308:] = 0

        boundary_points = rdl.sample_femur_interior_pts(boundary_mask,
                                                        N_lns=n_lines)
        boundary_points = rdl.forward_fill_jagged(boundary_points)

        # Tip: use midpoint-boundary estimator on [0.1, 0.5]
        s, e = tip_range
        tip_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        tip_bndry = rdl.forward_fill_jagged(tip_bndry)
        femur_tip = rdl.get_centroid_pts(tip_bndry)
        femur_tip = rdl.smooth_points(femur_tip, smooth_window)

        # Midpoint: midpoint-boundary estimator on [0.6, 1.0]
        s, e = midpoint_range
        midpt_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points,
                                                            s, e)
        midpt_bndry = rdl.forward_fill_jagged(midpt_bndry)
        femur_midpt = rdl.get_centroid_pts(midpt_bndry)
        femur_midpt = rdl.smooth_points(femur_midpt, smooth_window)

        radial_labels = rdl.label_radial_masks(mask, femur_tip, femur_midpt,
                                               N=n_segments)

        if inplace:
            self.radial_mask = radial_labels
            self.femur_tip = femur_tip
            self.femur_midpt = femur_midpt

        return radial_labels


if __name__ == "__main__":
    # Input path and naming consistent with other scripts (adjust if needed)
    pipeline = Aging1342Segmentation(
        input_path="data/raw/right 10 min-regional movement_00001342.npy",  # or updated path
        video_id="1342",
        condition="aging",
        n_segments=64,
    )
    pipeline.run(debug=True)
