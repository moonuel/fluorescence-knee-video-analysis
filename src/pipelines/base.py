import os
import numpy as np
from functools import partial
from src.utils import io, utils, views
from src.core import knee_segmentation as ks
from src.core import radial_segmentation as rdl

class KneeSegmentationPipeline:
    """
    Base class for the Knee Segmentation Pipeline.

    This class encapsulates the standard workflow for segmenting knee fluorescence videos,
    including preprocessing, mask generation (outer/inner/femur), and radial segmentation.

    How to Use:
    -----------
    1. Create a subclass of `KneeSegmentationPipeline` for your specific dataset/video type.
    2. In `__init__`, call `super().__init__(input_path)` and then override default parameters
       (e.g., `self.crop_size`, `self.otsu_scale`, etc.) to suit your data.
    3. Override the `preprocess()` method if you need custom rotation, cropping, or brightness adjustments.
    4. Override the `apply_manual_cuts(mask)` method to implement specific mask refinements (e.g., zeroing out artifacts).
    5. Instantiate your subclass and call `.run(video_id)`.

    Key Attributes (can be modified in subclass `__init__`):
    -------------------------------------------------------
    Preprocessing:
        crop_size (int): Size of the square crop (default 500).
        rot90_k (int): Number of 90-deg rotations (default 0).

    Mask Generation:
        blur_kernel (tuple): Kernel for initial blur (default (25, 25)).
        otsu_scale (float): Scaling factor for Otsu threshold (default 0.5).
        adaptive_block (int): Block size for adaptive thresholding (default 141).
        adaptive_c (int): Constant C for adaptive thresholding (default 10).

    Refinement:
        morph_erode_kernel, morph_open_kernel, morph_close_kernel: Kernels for morphological ops.

    Radial Segmentation:
        radial_n_lines (int): Number of scan lines (default 128).
        radial_n_segments (int): Number of output segments (default 64).
        tip_range (tuple): Range for tip estimation (default (0.05, 5))).
        midpoint_range (tuple): Range for midpoint estimation (default (0.6, 0.95)).

    Example:
    --------
    class MyPipeline(KneeSegmentationPipeline):
        def __init__(self, input_path):
            super().__init__(input_path)
            self.crop_size = 600
            self.otsu_scale = 0.6

        def apply_manual_cuts(self, mask):
            mask[:, :100, :] = 0  # Remove top strip
            return mask

    pipeline = MyPipeline("path/to/video.npy")
    pipeline.run(1234)
    """
    def __init__(self, input_path, output_dir=None, preview=True):
        self.input_path = input_path
        self.preview = preview
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default to data/processed relative to project root?
            # Or ../data/processed relative to script?
            # Let's assume input_path is relative or absolute, output next to it or specified.
            # We'll use a default if None.
            self.output_dir = os.path.join(os.path.dirname(input_path), '..', 'processed') # Assuming input in raw or similar?
            # Actually, standard is data/processed.
            
        self.video = None
        self.femur_mask = None
        self.outer_mask = None
        self.radial_masks = None
        
        # Default Parameters
        # --- Preprocessing ---
        self.crop_size = 500        # Size of the square crop centered on the frame
        self.rot90_k = 0            # Number of 90-degree counter-clockwise rotations
        
        # --- Mask Generation (Blurring) ---
        self.blur_kernel = (25, 25) # Kernel size for initial Gaussian blur
        
        # --- Mask Generation (Otsu Thresholding - Outer Mask) ---
        self.otsu_scale = 0.5       # Scaling factor for Otsu threshold
        
        # --- Mask Generation (Adaptive Thresholding - Inner Mask) ---
        self.adaptive_block = 141   # Block size for adaptive thresholding
        self.adaptive_c = 10        # Constant C for adaptive thresholding
        
        # --- Mask Refinement (Morphological Operations) ---
        self.morph_erode_kernel = None      # Kernel for erosion on outer mask (if needed)
        self.morph_open_kernel = (11, 11)   # Kernel for opening (remove noise)
        self.morph_close_kernel = (27, 27)  # Kernel for closing (fill holes)
        
        # --- Radial Segmentation ---
        self.radial_n_lines = 128           # Number of scan lines for boundary sampling
        self.radial_n_segments = 64         # Number of radial segments
        self.tip_range = (0.05, 0.5)        # Range for femur tip estimation
        self.midpoint_range = (0.6, 0.95)   # Range for femur midpoint estimation
        self.smooth_window = 9              # Window size for smoothing landmarks

    def load_data(self):
        print(f"Loading video from {self.input_path}...")
        self.video = io.load_nparray(self.input_path)

    def preprocess(self):
        print("Preprocessing video...")
        if self.rot90_k != 0:
            self.video = np.rot90(self.video, k=self.rot90_k, axes=(1, 2))
        
        self.video = utils.crop_video_square(self.video, self.crop_size)

    def match_histograms(self, video_blr):
        # Default behavior: match to first frame or don't match if not needed?
        # Most scripts match to a reference frame.
        # Subclasses can override to set a specific reference frame index.
        return video_blr

    def get_outer_mask(self, video_hist):
        print("Generating outer mask...")
        otsu_fn = partial(ks.get_otsu_masks, thresh_scale=self.otsu_scale)
        if hasattr(utils, 'parallel_process_video'):
            outer_mask = utils.parallel_process_video(video_hist, otsu_fn, batch_size=150)
        else:
            outer_mask = otsu_fn(video_hist)
            
        if self.morph_erode_kernel:
            outer_mask = utils.morph_erode(outer_mask, self.morph_erode_kernel)
            
        return outer_mask

    def get_inner_mask(self, video_hist):
        print("Generating inner mask...")
        return ks.mask_adaptive(video_hist, self.adaptive_block, self.adaptive_c)

    def apply_manual_cuts(self, mask):
        """Override this method in subclasses to apply specific cuts."""
        return mask

    def generate_femur_mask(self):
        print("Generating femur mask...")
        video_blr = utils.blur_video(self.video, self.blur_kernel)
        video_hist = self.match_histograms(video_blr)
        
        self.outer_mask = self.get_outer_mask(video_hist)
        inner_mask = self.get_inner_mask(video_hist)
        
        femur_mask = rdl.interior_mask(self.outer_mask, inner_mask)
        
        if self.morph_open_kernel:
            femur_mask = utils.morph_open(femur_mask, self.morph_open_kernel)
            
        if self.morph_close_kernel:
            femur_mask = utils.morph_close(femur_mask, self.morph_close_kernel)
            
        self.femur_mask = self.apply_manual_cuts(femur_mask)

    def segment_radial(self):
        print("Performing radial segmentation...")
        boundary_points = rdl.sample_femur_interior_pts(self.femur_mask, N_lns=self.radial_n_lines)
        if hasattr(rdl, 'forward_fill_jagged'):
            boundary_points = rdl.forward_fill_jagged(boundary_points)
            
        # Tip
        s, e = self.tip_range
        tip_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        if hasattr(rdl, 'forward_fill_jagged'):
            tip_bndry = rdl.forward_fill_jagged(tip_bndry)
        femur_tip = rdl.get_centroid_pts(tip_bndry)
        femur_tip = rdl.smooth_points(femur_tip, self.smooth_window)
        
        # Midpoint
        s, e = self.midpoint_range
        midpt_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        if hasattr(rdl, 'forward_fill_jagged'):
            midpt_bndry = rdl.forward_fill_jagged(midpt_bndry)
        femur_midpt = rdl.get_centroid_pts(midpt_bndry)
        femur_midpt = rdl.smooth_points(femur_midpt, self.smooth_window)
        
        self.radial_masks = rdl.label_radial_masks(self.outer_mask, femur_tip, femur_midpt, N=self.radial_n_segments)

    def save_results(self, video_id):
        if self.preview:
            response = input("Save results? (y/n): ")
            if response.lower() != 'y':
                print("Aborting save.")
                return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        vid_path = os.path.join(self.output_dir, f"{video_id}_radial_video_N{self.radial_n_segments}.npy")
        mask_path = os.path.join(self.output_dir, f"{video_id}_radial_masks_N{self.radial_n_segments}.npy")
        
        print(f"Saving video to {vid_path}")
        io.save_nparray(self.video, vid_path)
        print(f"Saving masks to {mask_path}")
        io.save_nparray(self.radial_masks, mask_path)

    def run(self, video_id):
        self.load_data()
        self.preprocess()
        self.generate_femur_mask()
        self.segment_radial()
        
        if self.preview:
            print("Previewing results...")
            views.show_frames([self.radial_masks * (255 // self.radial_n_segments), self.video])
            
        self.save_results(video_id)
        print("Pipeline finished.")
