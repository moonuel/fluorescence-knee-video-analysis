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
        video = super().preprocess(video, rot90_k, rot_angle, crop_size,
                                  empty_fill_value, inplace=True)

        # Create a SEPARATE clipped copy for segmentation only
        self.processed_video_clipped = self.processed_video.copy()
        self.processed_video_clipped[self.processed_video_clipped > 165] = 165

        return self.processed_video

    # ------------------------
    # Otsu / Outer Mask
    # ------------------------
    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25),
                           thresh_scale=0.7, hist_frame=142, inplace=False):
        video = self.processed_video_clipped
        return super().generate_otsu_mask(video, blur_kernel,
                                          thresh_scale, hist_frame, inplace)

    def refine_otsu_mask(self, mask=None, morph_open_kernel=None,
                         morph_close_kernel=(45, 45),
                         morph_erode_kernel=(25, 25),
                         morph_dilate_kernel=(25, 25), inplace=False):
        return super().refine_otsu_mask(mask, morph_open_kernel,
                                        morph_close_kernel, morph_erode_kernel,
                                        morph_dilate_kernel, inplace)

    # ------------------------
    # Interior / Femur Mask
    # ------------------------
    def generate_interior_mask(self, hist_video=None, adaptive_block=161,
                               adaptive_c=11, inplace=False):
        return super().generate_interior_mask(hist_video, adaptive_block,
                                              adaptive_c, inplace)

    def refine_femur_mask(self, mask=None, morph_open_kernel=(3,3),
                          morph_close_kernel=(11,11), morph_erode_kernel=None,
                          morph_dilate_kernel=None, inplace=False):
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
        """Run segmentation with femur-based derotation.

        1. Run full segmentation once to get femur tip/midpt
        2. Rotate video artifacts (processed_video, otsu_mask, femur_mask) using femur line
        3. Re-run only radial segmentation on rotated geometry

        Parameters
        ----------
        debug : bool, default=False
            If True, show intermediate results
        debug_pause : bool, default=False
            If True, pause between debug steps
        """
        print("=== Normal1207 Segmentation with Femur Derotation ===\n")

        # ------------------------
        # STEP 1: Full segmentation to estimate femur orientation
        # ------------------------
        print("Step 1: Running full segmentation to estimate femur orientation...")

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
        # STEP 2: Derotate video and masks about femur tip
        # ------------------------
        print("Step 2: Derotating video about femur tip to flatten femur line...")

        # Derotate video artifacts that will affect radial segmentation
        derotated_video = self._derotate_about_femur(self.processed_video)
        derotated_video[derotated_video==0] = 17
        self.processed_video = derotated_video

        # Derotate masks that define the segmentation geometry
        if self.otsu_mask is not None:
            self.otsu_mask = self._derotate_about_femur(self.otsu_mask.astype(np.uint8)).astype(bool)
        if self.femur_mask is not None:
            self.femur_mask = self._derotate_about_femur(self.femur_mask.astype(np.uint8)).astype(bool)

        # Recreate clipped version for segmentation stability
        self.processed_video_clipped = self.processed_video.copy()
        self.processed_video_clipped[self.processed_video_clipped > 165] = 165

        # Clear radial segmentation results (will recompute)
        self.radial_mask = None
        self.femur_tip = None
        self.femur_midpt = None

        # ------------------------
        # STEP 3: Re-run radial segmentation on derotated geometry
        # ------------------------
        print("Step 3: Re-running radial segmentation on derotated geometry...")

        self.radial_segmentation(inplace=True)
        self._show_radial_preview()
        if debug_pause: input("Press Enter to continue...")

        print("\n=== Segmentation Complete ===\n")

        # Save results (ask for confirmation in debug mode, auto-save in production)
        if debug:
            save_response = input("Debug mode: Save results? (y/n): ").strip().lower()
            if save_response == 'y':
                self.save_results(self.processed_video, self.radial_mask, self.femur_mask,
                                self.video_id, self.condition, self.n_segments)
            else:
                print("Debug mode: Save cancelled.")

        else:
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
    pipeline.run(debug=True)
    # pipeline.display_saved_results()
