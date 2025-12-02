import os
from pathlib import Path
import numpy as np
from functools import partial
from src.utils import io, utils, views
from src.core import knee_segmentation as ks
from src.core import radial_segmentation as rdl


# -------------------------------------------------------------------------
# Configuration Containers
# -------------------------------------------------------------------------

class PreprocessConfig:
    def __init__(self, crop_size=500, rot90_k=0):
        self.crop_size = crop_size
        self.rot90_k = rot90_k


class MaskConfig:
    def __init__(
        self,
        blur_kernel=(25, 25),
        otsu_scale=0.5,
        adaptive_block=141,
        adaptive_c=10,
        morph_erode_kernel=(7, 7),
        morph_open_kernel=(9, 9),
        morph_close_kernel=(13, 13),
    ):
        self.blur_kernel = blur_kernel
        self.otsu_scale = otsu_scale
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.morph_erode_kernel = morph_erode_kernel
        self.morph_open_kernel = morph_open_kernel
        self.morph_close_kernel = morph_close_kernel


class RadialConfig:
    def __init__(
        self,
        n_lines=128,
        n_segments=64,
        tip_range=(0.05, 0.5),
        midpoint_range=(0.6, 0.95),
        smooth_window=9
    ):
        self.n_lines = n_lines
        self.n_segments = n_segments
        self.tip_range = tip_range
        self.midpoint_range = midpoint_range
        self.smooth_window = smooth_window


# -------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------

class KneeSegmentationPipeline:
    """
    Base class for the full segmentation workflow:
        - load → preprocess → masks → radial segmentation → save
    Subclasses override configuration or specific methods.
    """

    def __init__(self, input_path, output_dir=None, preview=True):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.default_output_dir()
        self.preview = preview

        # Configuration blocks
        self.pre_cfg = PreprocessConfig()
        self.mask_cfg = MaskConfig()
        self.radial_cfg = RadialConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # internal holders
        self.video_hist = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def default_output_dir(self):
        """
        Default: store alongside the input file in a `processed/` directory.
        """
        parent = self.input_path.parent
        return parent / "processed"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_video(self):
        vid = io.load_nparray(self.input_path)
        if vid.ndim != 3:
            raise ValueError(f"Expected (T, H, W) video array; got shape {vid.shape}")
        return vid

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, video):
        """
        Default preprocessing: rotation → center crop.
        Subclasses can override for brightness, denoising, etc.
        """
        if self.pre_cfg.rot90_k:
            video = np.rot90(video, k=self.pre_cfg.rot90_k, axes=(1, 2))

        return utils.crop_video_square(video, self.pre_cfg.crop_size)

    # ------------------------------------------------------------------
    # Mask Generation
    # ------------------------------------------------------------------

    def generate_otsu_mask(self, video):
        """
        Compute per-frame binary outer masks using Otsu (parallelized if available).
        This method stores a blurred / preprocessed video in self.video_hist for later use.
        """
        print("Generating Otsu mask... (Default: blur + hist match + Otsu)")

        mk = self.mask_cfg

        # Blur video (kept for later inner-mask computations)
        video_blr = utils.blur_video(video, mk.blur_kernel)
        # matching step is optional in subclasses; default: use blurred video as-is
        video_hist = video_blr
        self.video_hist = video_hist

        otsu_fn = partial(ks.get_otsu_masks, thresh_scale=mk.otsu_scale)
        if hasattr(utils, "parallel_process_video"):
            outer_mask = utils.parallel_process_video(video_hist, otsu_fn, batch_size=150)
        else:
            outer_mask = otsu_fn(video_hist)

        return outer_mask
    
    def refine_otsu_mask(self, mask):
        """
        Override this to perform manual refinements to the Otsu mask. 
        Default: passthrough
        """
        print("Refining Otsu mask... (Default: passthrough)")
        return mask

    # ------------------------------------------------------------------
    # Refinement / Manual Cuts
    # ------------------------------------------------------------------

    def refine_femur_mask(self, mask):
        """
        Override this to remove artifacts or enforce boundaries, e.g. manual cuts or morph operations
        Default: passthrough.
        """
        print("Applying refinements to the femur mask... (Default: passthrough)")
        return mask

    # ------------------------------------------------------------------
    # Femur Mask
    # ------------------------------------------------------------------

    def generate_femur_mask(self, outer_mask):
        """
        Combine outer mask with adaptive inner mask to produce femur_mask,
        then apply morphological cleanup.
        """
        print("Generating femur mask... (Default: get interior of inner_mask using outer_mask)")
        mk = self.mask_cfg
        if self.video_hist is None:
            # defensive: if generate_masks wasn't called, try to compute a simple inner mask fallback
            inner_mask = ks.mask_adaptive(outer_mask, mk.adaptive_block, mk.adaptive_c)
        else:
            inner_mask = ks.mask_adaptive(self.video_hist, mk.adaptive_block, mk.adaptive_c)

        femur_mask = rdl.interior_mask(outer_mask, inner_mask)

        if mk.morph_open_kernel:
            femur_mask = utils.morph_open(femur_mask, mk.morph_open_kernel)

        if mk.morph_close_kernel:
            femur_mask = utils.morph_close(femur_mask, mk.morph_close_kernel)

        return femur_mask

    # ------------------------------------------------------------------
    # Radial Segmentation
    # ------------------------------------------------------------------

    def radial_segmentation(self, mask, femur_mask):
        """
        Extract radial lines from the femur tip through joint space and label radial segments.
        """
        rc = self.radial_cfg

        boundary_points = rdl.sample_femur_interior_pts(femur_mask, N_lns=rc.n_lines)
        boundary_points = rdl.forward_fill_jagged(boundary_points)

        # Tip
        s, e = rc.tip_range
        tip_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        tip_bndry = rdl.forward_fill_jagged(tip_bndry)
        femur_tip = rdl.get_centroid_pts(tip_bndry)
        femur_tip = rdl.smooth_points(femur_tip, rc.smooth_window)

        # Midpoint
        s, e = rc.midpoint_range
        midpt_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        midpt_bndry = rdl.forward_fill_jagged(midpt_bndry)
        femur_midpt = rdl.get_centroid_pts(midpt_bndry)
        femur_midpt = rdl.smooth_points(femur_midpt, rc.smooth_window)

        radial_labels = rdl.label_radial_masks(mask, femur_tip, femur_midpt, N=rc.n_segments)
        return radial_labels

    # ------------------------------------------------------------------
    # Saving / Output
    # ------------------------------------------------------------------

    def confirm_save(self):
        """
        Hook for requesting user permission before saving results.
        Default: always save. Subclasses or callers may override.
        """
        if not self.preview:
            return True  # no interactive prompts when preview disabled

        resp = input("Save results? (y/n): ").strip().lower()
        return resp == "y"

    def save_results(self, masks, femur_mask, radial_labels, video_id):
        io.save_nparray(masks, self.output_dir / f"{video_id}_mask.npy")
        io.save_nparray(femur_mask, self.output_dir / f"{video_id}_femur.npy")
        io.save_nparray(radial_labels, self.output_dir / f"{video_id}_radial.npy")

    # ------------------------------------------------------------------
    # Pipeline Entry Point
    # ------------------------------------------------------------------

    def run(self, video_id=None):
        video = self.load_video()
        video = self.preprocess(video)

        masks = self.generate_otsu_mask(video)
        masks = self.refine_otsu_mask(masks)

        femur = self.generate_femur_mask(masks)
        femur = self.refine_femur_mask(femur)

        radial = self.radial_segmentation(masks, femur)

        if self.preview:
            views.show_frames([radial * (255 // self.radial_cfg.n_segments), video])

        if self.confirm_save():
            self.save_results(masks, femur, radial, video_id or "result")
        else:
            print("Save cancelled.")

        return masks, femur, radial

