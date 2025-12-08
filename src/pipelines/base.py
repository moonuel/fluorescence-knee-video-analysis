import os
from pathlib import Path
import numpy as np
from functools import partial
from utils import io, utils, views
from core import knee_segmentation as ks
from core import radial_segmentation as rdl


# -------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------

class KneeSegmentationPipeline:
    """
    Base class for knee segmentation workflow with flexible debugging.
    
    Architecture:
    -------------
    - All pipeline methods are parameterized with sensible defaults
    - Methods support both functional and inplace operation modes
    - Subclasses override methods to provide optimized parameters
    - Single run() method handles both debugging and production
    
    Workflow:
    ---------
    1. Load video data
    2. Preprocess (rotation, crop, fill)
    3. Generate Otsu mask (outer boundary)
    4. Refine Otsu mask (morphological operations)
    5. Generate interior mask (adaptive thresholding)
    6. Generate femur mask (intersection of outer and interior)
    7. Refine femur mask (morphological operations, manual cuts)
    8. Radial segmentation (partition into angular segments)
    
    Usage Patterns:
    ---------------
    
    1. **Override methods in subclass with optimized parameters:**
    
        class MySegmentation(KneeSegmentationPipeline):
            def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), 
                                  thresh_scale=0.6, hist_frame=50, inplace=False):
                return super().generate_otsu_mask(video, blur_kernel, 
                                                 thresh_scale, hist_frame, inplace)
    
    2. **Debug mode - visualize all intermediate steps:**
    
        pipeline = MySegmentation(input_path)
        pipeline.run("output_id", debug=True)
        
        # Or with step-by-step inspection:
        pipeline.run("output_id", debug=True, debug_pause=True)
    
    3. **Production mode - clean execution:**
    
        pipeline = MySegmentation(input_path)
        pipeline.run("output_id")  # Uses all overridden parameters
    
    Key Features:
    -------------
    - **Inplace operations**: Methods update self.* attributes and return results
    - **State maintenance**: Pipeline maintains intermediate results for inspection
    - **Debug visualization**: Show intermediate steps without separate test scripts
    - **Parameter optimization**: Test parameters via method overrides, then run()
    - **Consistent workflow**: Debug and production use identical code paths
    
    Attributes (Intermediate Results):
    -----------------------------------
    video : ndarray
        Original loaded video
    processed_video : ndarray
        After preprocessing
    video_hist : ndarray
        Histogram-matched blurred video (for interior mask generation)
    otsu_mask : ndarray
        Outer boundary mask
    interior_mask : ndarray
        Inner boundary mask
    femur_mask : ndarray
        Final femur region mask
    radial_mask : ndarray
        Radial segmentation labels
    """

    def __init__(self, input_path, video_id, condition, n_segments=64, output_dir=None):
        """
        Initialize the knee segmentation pipeline.

        Parameters:
        -----------
        input_path : str or Path
            Path to the input video file (.npy format)
        video_id : str
            Unique identifier for this video (e.g., "1357", "0308")
        condition : str
            Experimental condition (e.g., "normal", "aging", "dmm")
        n_segments : int, default=64
            Number of radial segments for segmentation
        output_dir : str or Path, optional
            Directory to save processed results. If None, uses default location.
        """
        self.input_path = Path(input_path)
        self.video_id = video_id
        self.condition = condition
        self.n_segments = n_segments
        self.output_dir = Path(output_dir) if output_dir else self.default_output_dir()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # internal holders
        self.video = self.load_video()
        self.processed_video = None
        self.video_hist = None
        self.otsu_mask = None
        self.interior_mask = None
        self.femur_mask = None
        self.radial_mask = None
        print(f"Initialized segmentation pipeline: {condition}_{video_id} (N={n_segments})")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def default_output_dir(self):
        """
        Default: store in the data/segmented directory.
        """
        return Path("data/segmented")

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

    def preprocess(self, video=None, rot90_k=1, rot_angle = None, crop_size=500, empty_fill_value=None, inplace=False):
        """
        Default preprocessing: rotation → center crop → empty fill.
        Subclasses can override for brightness, denoising, etc.
        """
        if video is None:
            video = self.video

        if rot90_k:
            video = np.rot90(video, k=rot90_k, axes=(1, 2))

        if rot_angle:
            video = utils.rotate_video(video, rot_angle)

        video = utils.center_crop(video, crop_size)

        if empty_fill_value is not None:
            video[video == 0] = empty_fill_value

        if inplace:
            self.processed_video = video

        return video

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def view(self):
        views.show_frames(self.video)

    def view_processed(self):
        if self.processed_video is not None:
            views.show_frames(self.processed_video)
        else:
            print("No processed video available. Call preprocess() first.")

    def view_otsu_mask(self):
        if self.otsu_mask is not None:
            views.show_frames(self.otsu_mask)
        else:
            print("No Otsu mask available. Call generate_otsu_mask() first.")

    def view_femur_mask(self):
        if self.femur_mask is not None:
            views.show_frames(self.femur_mask)
        else:
            print("No femur mask available. Call generate_femur_mask() first.")

    def view_radial_mask(self):
        if self.radial_mask is not None:
            views.show_frames(self.radial_mask)
        else:
            print("No radial mask available. Call radial_segmentation() first.")

    # ------------------------------------------------------------------
    # Mask Generation
    # ------------------------------------------------------------------

    def generate_otsu_mask(self, video=None, blur_kernel=(25, 25), thresh_scale=0.5, hist_frame = 0, inplace=False):
        """
        Compute per-frame binary outer masks using Otsu (parallelized if available).
        This method stores a blurred / preprocessed video in self.video_hist for later use.
        """
        print("Generating Otsu mask... (Default: blur + hist match + Otsu)")

        if video is None:
            video = self.processed_video if self.processed_video is not None else self.video

        # Blur video (kept for later inner-mask computations)
        video_blr = utils.blur_video(video, blur_kernel)
        # matching step is optional in subclasses; default: use blurred video as-is
        video_hist = rdl.match_histograms_video(video_blr, video_blr[hist_frame])

        if inplace:
            self.video_hist = video_hist

        otsu_fn = partial(ks.get_otsu_masks, thresh_scale=thresh_scale)
        if hasattr(utils, "parallel_process_video"):
            outer_mask = utils.parallel_process_video(video_hist, otsu_fn, batch_size=150)
        else:
            outer_mask = otsu_fn(video_hist)

        if inplace:
            self.otsu_mask = outer_mask

        return outer_mask
    
    def refine_otsu_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        """
        Override this to perform manual refinements to the Otsu mask.
        Apply morphological operations in sequence: erode → dilate → open → close.
        Default: passthrough
        """
        print("Refining Otsu mask... (Default: passthrough)")
        if mask is None:
            mask = self.otsu_mask

        # Apply morphological operations in sequence
        if morph_open_kernel:
            mask = utils.morph_open(mask, morph_open_kernel)

        if morph_close_kernel:
            mask = utils.morph_close(mask, morph_close_kernel)

        if morph_erode_kernel:
            mask = utils.morph_erode(mask, morph_erode_kernel)

        if morph_dilate_kernel:
            mask = utils.morph_dilate(mask, morph_dilate_kernel)

        if inplace:
            self.otsu_mask = mask

        return mask

    # ------------------------------------------------------------------
    # Interior Mask Generation
    # ------------------------------------------------------------------

    def generate_interior_mask(self, hist_video=None, adaptive_block=141, adaptive_c=10, inplace=False):
        """
        Generate interior mask using adaptive thresholding on the blurred video histogram.
        This creates the inner boundary that will be combined with the outer mask.
        """
        print("Generating interior mask... (Default: adaptive thresholding)")
        if hist_video is None:
            print("Default: computing interior mask from histogram")
            hist_video = self.video_hist

        if hist_video is None:
            raise ValueError("No histogram-matched video available. Call generate_otsu_mask() first.")

        inner_mask = ks.mask_adaptive(hist_video, adaptive_block, adaptive_c)

        if inplace:
            self.interior_mask = inner_mask

        return inner_mask

    # ------------------------------------------------------------------
    # Refinement / Manual Cuts
    # ------------------------------------------------------------------

    def refine_femur_mask(self, mask=None, morph_open_kernel=None, morph_close_kernel=None, morph_erode_kernel=None, morph_dilate_kernel=None, inplace=False):
        """
        Override this to perform manual refinements to the femur mask.
        Apply morphological operations and manual cuts in sequence.
        Default: passthrough.
        """
        print("Refining femur mask... (Default: passthrough)")
        if mask is None:
            mask = self.femur_mask

        # Apply morphological operations in sequence
        if morph_open_kernel:
            mask = utils.morph_open(mask, morph_open_kernel)

        if morph_close_kernel:
            mask = utils.morph_close(mask, morph_close_kernel)

        if morph_erode_kernel:
            mask = utils.morph_erode(mask, morph_erode_kernel)

        if morph_dilate_kernel:
            mask = utils.morph_dilate(mask, morph_dilate_kernel)

        if inplace:
            self.femur_mask = mask

        return mask

    # ------------------------------------------------------------------
    # Femur Mask
    # ------------------------------------------------------------------

    def generate_femur_mask(self, outer_mask=None, interior_mask=None, inplace=False):
        """
        Combine outer mask with interior mask to produce femur_mask.
        """
        print("Generating femur mask... (Default: intersect outer and interior masks)")
        if outer_mask is None:
            outer_mask = self.otsu_mask
        if interior_mask is None:
            interior_mask = self.interior_mask

        femur_mask = rdl.interior_mask(outer_mask, interior_mask)

        if inplace:
            self.femur_mask = femur_mask

        return femur_mask

    # ------------------------------------------------------------------
    # Radial Segmentation
    # ------------------------------------------------------------------

    def radial_segmentation(self, mask=None, femur_mask=None, n_lines=128, n_segments=64, tip_range=(0.05, 0.5), midpoint_range=(0.6, 0.95), smooth_window=9, inplace=False):
        """
        Extract radial lines from the femur tip through joint space and label radial segments.
        """
        if mask is None:
            mask = self.otsu_mask
        if femur_mask is None:
            femur_mask = self.femur_mask

        boundary_points = rdl.sample_femur_interior_pts(femur_mask, N_lns=n_lines)
        boundary_points = rdl.forward_fill_jagged(boundary_points)

        # Tip
        s, e = tip_range
        tip_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        tip_bndry = rdl.forward_fill_jagged(tip_bndry)
        femur_tip = rdl.get_centroid_pts(tip_bndry)
        femur_tip = rdl.smooth_points(femur_tip, smooth_window)

        # Midpoint
        s, e = midpoint_range
        midpt_bndry = rdl.estimate_femur_midpoint_boundary(boundary_points, s, e)
        midpt_bndry = rdl.forward_fill_jagged(midpt_bndry)
        femur_midpt = rdl.get_centroid_pts(midpt_bndry)
        femur_midpt = rdl.smooth_points(femur_midpt, smooth_window)

        radial_labels = rdl.label_radial_masks(mask, femur_tip, femur_midpt, N=n_segments)

        if inplace:
            self.radial_mask = radial_labels

        return radial_labels

    # ------------------------------------------------------------------
    # Saving / Output
    # ------------------------------------------------------------------

    def display_saved_results(self):
        """
        Load and display previously saved segmentation results.

        Looks for saved files in the output directory using the standardized naming convention:
        {condition}_{video_id}_{datatype}_N{n_segments}.npy

        Displays:
        - Processed video
        - Femur mask
        - Radial segmentation mask (color-coded)

        Raises:
        -------
        FileNotFoundError
            If any of the required saved files are not found
        """
        # Build expected filenames
        video_filename = f"{self.condition}_{self.video_id}_video_N{self.n_segments}.npy"
        radial_filename = f"{self.condition}_{self.video_id}_radial_N{self.n_segments}.npy"
        femur_filename = f"{self.condition}_{self.video_id}_femur_N{self.n_segments}.npy"

        video_path = self.output_dir / video_filename
        radial_path = self.output_dir / radial_filename
        femur_path = self.output_dir / femur_filename

        # Check if files exist
        missing_files = []
        if not video_path.exists():
            missing_files.append(video_filename)
        if not radial_path.exists():
            missing_files.append(radial_filename)
        if not femur_path.exists():
            missing_files.append(femur_filename)

        if missing_files:
            raise FileNotFoundError(f"Missing saved result files: {', '.join(missing_files)} in {self.output_dir}")

        # Load the saved results
        print(f"Loading saved results for {self.condition}_{self.video_id}...")
        saved_video = io.load_nparray(video_path)
        saved_radial = io.load_nparray(radial_path)
        saved_femur = io.load_nparray(femur_path)

        # TEMP:
        # saved_radial = (saved_radial // 4)*4

        # Display the results
        print("Displaying saved segmentation results...")
        views.show_frames([
            saved_video,
            # saved_femur,
            saved_radial * (255 // self.n_segments),  # Color-code radial segments
            views.draw_mask_boundary(saved_video, saved_radial)  # Overlay boundaries
        ])

    def confirm_save(self):
        """
        Hook for requesting user permission before saving results.
        Default: always save. Subclasses or callers may override.
        """
        return True  # Always save by default

    def save_results(self, video, radial_mask, femur_mask, video_id, condition, n_segments):
        """
        Save segmentation results using standardized naming convention.
        
        Naming convention: {condition}_{video_id}_{datatype}_N{segments}.npy
        
        Parameters:
        -----------
        video : ndarray
            Preprocessed video
        radial_mask : ndarray
            Radial segmentation mask
        femur_mask : ndarray
            Femur region mask
        video_id : str
            Video identifier (e.g., "1357", "0308")
        condition : str, optional
            Experimental condition (e.g., "normal", "aging", "dmm")
            If None, will default to "unknown"
        n_segments : int, default=64
            Number of radial segments used (for radial mask filename)
        """
        print(f"Saving to {self.output_dir}...")
        
        # Default condition if not provided
        if condition is None:
            condition = "unknown"
        
        # Build filenames using new convention
        video_filename = f"{condition}_{video_id}_video_N{n_segments}.npy"
        radial_filename = f"{condition}_{video_id}_radial_N{n_segments}.npy"
        femur_filename = f"{condition}_{video_id}_femur_N{n_segments}.npy"
        
        # Save files
        print("Saving video...")
        io.save_nparray(video, self.output_dir / video_filename)
        print("Saving radial masks...")
        io.save_nparray(radial_mask, self.output_dir / radial_filename)
        print("Saving femur masks...")
        io.save_nparray(femur_mask, self.output_dir / femur_filename)
        
        print(f"Saved: {video_filename}, {radial_filename}, {femur_filename}")

    # ------------------------------------------------------------------
    # Pipeline Entry Point
    # ------------------------------------------------------------------

    def run(self, debug=False, debug_pause=False):
        """
        Run the full segmentation pipeline.
        
        1. Preprocess
        2. Generate Otsu mask
        3. Refine Otsu mask
        4. Generate interior mask
        5. Generate femur mask
        6. Refine femur mask
        7. Radial segmentation

        Parameters:
        -----------
        debug : bool, default=False
            If True, displays intermediate results at each pipeline step.
            Useful for validating parameter choices and algorithm behavior.
        debug_pause : bool, default=False
            If True, waits for user input between debug visualizations.
            Only has effect when debug=True.

        Returns:
        --------
        processed_video : ndarray
            The preprocessed video
        radial_mask : ndarray
            The final radial segmentation mask

        Usage:
        ------
        # Debug mode: visualize all intermediate steps
        pipeline.run(debug=True)

        # Production mode: clean execution
        pipeline.run()

        # Step-by-step debugging
        pipeline.run(debug=True, debug_pause=True)
        """
        print("=== Starting Segmentation Pipeline ===\n")
        
        # Step 1: Preprocessing
        print("Step 1: Preprocessing...")
        self.preprocess(inplace=True)
        if debug:
            views.show_frames(self.processed_video, "1. Preprocessed Video")
            if debug_pause: input("Press Enter to continue...")

        # Step 2: Generate Otsu Mask
        print("Step 2: Generating Otsu mask...")
        self.generate_otsu_mask(inplace=True)
        if debug:
            views.show_frames(self.otsu_mask, "2. Otsu Mask (Outer Boundary)")
            if debug_pause: input("Press Enter to continue...")

        # Step 3: Refine Otsu Mask
        print("Step 3: Refining Otsu mask...")
        self.refine_otsu_mask(inplace=True)
        if debug:
            views.show_frames(self.otsu_mask, "3. Refined Otsu Mask")
            if debug_pause: input("Press Enter to continue...")

        # Step 4: Generate Interior Mask
        print("Step 4: Generating interior mask...")
        self.generate_interior_mask(inplace=True)
        if debug:
            views.show_frames(self.interior_mask, "4. Interior Mask")
            if debug_pause: input("Press Enter to continue...")

        # Step 5: Generate Femur Mask
        print("Step 5: Generating femur mask...")
        self.generate_femur_mask(inplace=True)
        if debug:
            views.show_frames(self.femur_mask, "5. Femur Mask (Intersection)")
            if debug_pause: input("Press Enter to continue...")

        # Step 6: Refine Femur Mask
        print("Step 6: Refining femur mask...")
        self.refine_femur_mask(inplace=True)
        if debug:
            views.show_frames(self.femur_mask, "6. Refined Femur Mask")
            if debug_pause: input("Press Enter to continue...")

        # Step 7: Radial Segmentation
        print("Step 7: Performing radial segmentation...")
        self.radial_segmentation(inplace=True)
        if debug:
            views.show_frames([self.radial_mask * (255 // 64), views.draw_mask_boundary(self.processed_video, self.radial_mask)], "7. Radial Segmentation")
            if debug_pause: input("Press Enter to continue...")

        print("\n=== Pipeline Complete ===\n")

        # Always show final results unless in debug mode (to avoid duplication)
        if not debug:
            views.show_frames([self.radial_mask * (255 // 64), self.femur_mask, self.processed_video],
                            "Final Results")

        # Save results (ask for confirmation in debug mode, auto-save in production)
        if debug:
            save_response = input("Debug mode: Save results? (y/n): ").strip().lower()
            if save_response == 'y':
                self.save_results(self.processed_video, self.radial_mask, self.femur_mask,
                                self.video_id, self.condition, self.n_segments)
            else:
                print("Debug mode: Save cancelled.")
        else:
            # Production mode: always save
            self.save_results(self.processed_video, self.radial_mask, self.femur_mask,
                            self.video_id, self.condition, self.n_segments)

        return self.processed_video, self.radial_mask
