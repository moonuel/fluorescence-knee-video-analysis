import numpy as np
import cv2
from pipelines.base import KneeSegmentationPipeline
from utils import utils, views


class Normal1207Segmentation(KneeSegmentationPipeline):
    """Segmentation pipeline for normal knee 1207 using the new interface.

    Mapped from legacy normal_1207_radial.py:
    - Preprocessing: center crop 500, rot90=1, rotate -27°, fill empty pixels with 17.
    - Otsu outer mask: histogram match to frame 142, thresh_scale=0.7,
      then close(45x45) + erode(25x25).
    - Interior mask: adaptive block=161, C=11.
    - Femur mask refinement: open(11x11) plus manual cuts on y/x matching
      the legacy cropping for both femur mask and boundary sampling.
    - Radial segmentation: base implementation with parameters tuned to
      legacy choices: n_lines=128, n_segments=64, tip range around 0.6,
      midpoint range (0.1, 0.5), smoothing window=9.
    """

    # ------------------------
    # Preprocessing
    # ------------------------
    def preprocess(self, video=None, rot90_k=1, rot_angle=-27, crop_size=500,
                   empty_fill_value=17, inplace=False):
        """Crop, rotate, fill empty borders, and create clipped copy for stable segmentation.

        Legacy mapping (load_1207_normal_video):
        - video = utils.center_crop(video, 500)
        - video = np.rot90(video, k=1, axes=(1, 2))
        - video = utils.rotate_video(video, -27)
        - video[video == 0] = 17

        Additional: create a separate clipped copy for segmentation
        - processed_video_clipped = processed_video.copy()
        - processed_video_clipped[processed_video_clipped > 165] = 165
        """
        video = super().preprocess(video, rot90_k, rot_angle, crop_size,
                                  empty_fill_value, inplace=True)

        # processed_video is the ORIGINAL preprocessed video (for saving)

        # Create a SEPARATE clipped copy for segmentation only
        self.processed_video_clipped = self.processed_video.copy()
        self.processed_video_clipped[self.processed_video_clipped > 165] = 165

        return self.processed_video

    # ------------------------
    # Otsu / Outer Mask
    # ------------------------
    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25),
                           thresh_scale=0.7, hist_frame=142, inplace=False):
        """Outer mask with histogram matching to frame 142 and thresh_scale=0.7.

        Uses the clipped video for more stable segmentation.

        Legacy mapping (get_femur_mask):
        - ref_fm = video[142]
        - match_histograms_video(..., reference_frame=ref_fm)
        - get_otsu_masks(..., thresh_scale=0.7)
        """
        if video is None:
            video = getattr(self, "processed_video_clipped", None)
            if video is None:
                # Fallback to base behavior if clipping wasn't created
                video = self.processed_video

        return super().generate_otsu_mask(video, blur_kernel,
                                          thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=None,
                         morph_close_kernel=(45, 45),
                         morph_erode_kernel=(25, 25),
                         morph_dilate_kernel=None, inplace=False):
        """Close then erode the Otsu mask.

        Legacy mapping:
        - outer_mask = morph_close(outer_mask, (45,45))
        - outer_mask = morph_erode(outer_mask, (25,25))
        """
        return super().refine_otsu_mask(mask, morph_open_kernel,
                                        morph_close_kernel, morph_erode_kernel,
                                        morph_dilate_kernel, inplace)

    # ------------------------
    # Interior / Femur Mask
    # ------------------------
    def generate_interior_mask(self, hist_video=None, adaptive_block=161,
                               adaptive_c=11, inplace=False):
        """Interior mask using adaptive thresholding.

        Legacy mapping:
        - inner_mask = ks.mask_adaptive(video_hist, 161, 11)
        """
        return super().generate_interior_mask(hist_video, adaptive_block,
                                              adaptive_c, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(3,3),
                          morph_close_kernel=(11,11), morph_erode_kernel=None,
                          morph_dilate_kernel=None, inplace=False):
        """Refine femur mask: open and manual cropping.

        Legacy refinements:
        - morph_open(femur_mask, (11,11))
        - femur_mask[:, 333:, :] = 0

        Additional mapping from get_boundary_points for better boundary
        sampling (pushed into femur mask itself so the base radial
        segmentation sees a similar mask):
        - mask[:, 329:, :] = 0   # cut y below 329
        - mask[:, :180, :] = 0   # cut y above 180
        - mask[:, :, 338:] = 0   # cut x beyond 338

        (These extra cuts slightly strengthen the cropping vs. the
        original femur mask but approximate the same region used for
        boundary sampling.)
        """
        if mask is None:
            mask = self.femur_mask

        # Manual cropping
        mask[:, 329:, :] = 0
        mask[:, :170, :] = 0
        mask[:, :, 350:] = 0

        return super().refine_femur_mask(mask, morph_open_kernel,
                                         morph_close_kernel, morph_erode_kernel,
                                         morph_dilate_kernel, inplace)

    # ------------------------
    # Radial Segmentation
    # ------------------------
    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128,
                            n_segments=64, tip_range=(0.05, 0.45),
                            midpoint_range=(0.6, 0.95), smooth_window=9,
                            inplace=False):
        """Radial segmentation with parameters tuned to legacy behavior.

        Legacy mapping:
        - N_lns=128
        - femur_tip from estimate_femur_tip_boundary(..., 0.6), smoothed window=9
        - femur_midpt from estimate_femur_midpoint_boundary(..., 0.1–0.5), window=9

        Approximations:
        - Base implementation uses midpoint-boundary estimator for the tip
          as well; tip_range is centered around 0.6 to keep location close.
        - Uses the pipeline Otsu mask for radial labels instead of a
          separate Otsu (hist_frame=175, thresh_scale=0.65) computed in
          the legacy main(). If you want that dual-mask behavior back, we
          can add a small custom override later.
        """
        return super().radial_segmentation(mask, femur_mask, n_lines,
                                           n_segments, tip_range,
                                           midpoint_range, smooth_window,
                                           inplace)

    # ------------------------
    # Femur-based Derotation
    # ------------------------
    def _derotate_about_femur(self, video):
        """Derotate video so that femur line (tip -> midpt) becomes horizontal.

        For each frame, rotates around femur_tip so that the line from
        femur_tip to femur_midpt becomes horizontal (0 degrees).

        Parameters
        ----------
        video : ndarray, shape (T, H, W)
            Input video to derotate

        Returns
        -------
        ndarray, shape (T, H, W)
            Derotated video
        """
        T, H, W = video.shape
        derotated = np.zeros_like(video)

        for t in range(T):
            # Coerce femur tip and midpoint into numeric (x,y) pairs
            # Handle jagged dtype=object arrays from radial segmentation
            tip_arr = np.asarray(self.femur_tip[t]).reshape(-1, 2)[0]
            midpt_arr = np.asarray(self.femur_midpt[t]).reshape(-1, 2)[0]

            x1, y1 = float(tip_arr[0]), float(tip_arr[1])
            x2, y2 = float(midpt_arr[0]), float(midpt_arr[1])

            # Calculate angle of femur line relative to horizontal
            dx = x2 - x1
            dy = y2 - y1
            theta_deg = np.degrees(np.arctan2(dy, dx))

            # Rotation needed to make line horizontal: -theta
            rotation_angle = theta_deg

            # Create rotation matrix around femur tip
            center = (x1, y1)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

            # Apply rotation
            derotated[t] = cv2.warpAffine(video[t], M, (W, H))

        return derotated

    # ------------------------
    # Two-pass Run with Derotation
    # ------------------------
    def run(self, debug=False, debug_pause=False):
        """Run two-pass segmentation with femur-based derotation.

        Pass 1: Standard segmentation to estimate femur tip/midpt
        Derotation: Rotate video so femur line becomes horizontal
        Pass 2: Full segmentation on derotated video

        Parameters
        ----------
        debug : bool, default=False
            If True, show intermediate results
        debug_pause : bool, default=False
            If True, pause between debug steps
        """
        print("=== Normal1207 Two-Pass Segmentation with Femur Derotation ===\n")

        # ------------------------
        # PASS 1: Get initial femur orientation
        # ------------------------
        print("Pass 1: Estimating femur orientation...")

        self.preprocess(inplace=True)
        if debug:
            self.view_processed()
            if debug_pause: input("Press Enter to continue...")

        self.generate_otsu_mask(inplace=True)
        if debug:
            self.view_otsu_mask()
            if debug_pause: input("Press Enter to continue...")

        self.refine_otsu_mask(inplace=True)
        self.generate_interior_mask(inplace=True)
        self.generate_femur_mask(inplace=True)
        self.refine_femur_mask(inplace=True)
        self.radial_segmentation(inplace=True)  # Populates femur_tip/midpt

        if debug:
            self.view_femur_mask()
            self.view_radial_mask()
            if debug_pause: input("Press Enter to continue...")

        # ------------------------
        # DEROTATION: Apply femur-based rotation
        # ------------------------
        print("Derotating video about femur tip to flatten femur line...")

        # Derotate the processed video
        derotated_video = self._derotate_about_femur(self.processed_video)
        self.processed_video = derotated_video

        # Recreate clipped version for segmentation stability
        self.processed_video_clipped = self.processed_video.copy()
        self.processed_video_clipped[self.processed_video_clipped > 165] = 165

        # Clear old masks to force recomputation
        self.otsu_mask = None
        self.interior_mask = None
        self.femur_mask = None
        self.radial_mask = None
        self.femur_tip = None
        self.femur_midpt = None

        # ------------------------
        # PASS 2: Full segmentation on derotated video
        # ------------------------
        print("Pass 2: Running segmentation on derotated video...")

        self.video[self.video==0] = 17 # Refill empty pixels
        self.generate_otsu_mask(inplace=True)
        if debug:
            self.view_otsu_mask()
            if debug_pause: input("Press Enter to continue...")

        self.refine_otsu_mask(inplace=True)
        self.generate_interior_mask(inplace=True)
        self.generate_femur_mask(inplace=True)
        if debug:
            self.view_femur_mask()
            if debug_pause: input("Press Enter to continue...")

        self.refine_femur_mask(inplace=True)
        self.radial_segmentation(inplace=True)
        if debug:
            self.view_radial_mask()
            if debug_pause: input("Press Enter to continue...")

        print("\n=== Segmentation Complete ===\n")

        # Display final results (unless debug mode, which already showed them)
        if not debug:
            # draw boundary between seg 1 and seg N on processed_video
            boundary_overlay = views.draw_boundary_line(
                self.processed_video,
                self.radial_mask,
                seg_num=1,
                n_segments=self.n_segments,
                show_video=False,
            )

            boundary_overlay = views.draw_outer_radial_mask_boundary(
                boundary_overlay, self.radial_mask)

            views.show_frames([
                self.radial_mask * (255 // self.n_segments),
                boundary_overlay
            ], "7. Radial Segmentation with Boundary Line")

        

        # Save results
        self.save_results(self.processed_video, self.radial_mask, self.femur_mask,
                         self.video_id, self.condition, self.n_segments)

        return self.processed_video, self.radial_mask


if __name__ == "__main__":
    pipeline = Normal1207Segmentation(
        input_path="data/raw/dmm-0 min-fluid movement_00001207.npy",
        video_id="1207",
        condition="normal",
        n_segments=64,
    )
    # pipeline.run(debug=True)
    pipeline.display_saved_results()
