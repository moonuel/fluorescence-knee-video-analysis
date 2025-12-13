#!/usr/bin/env python3
"""
Export segmentation previews for all 7 knee videos (4 normal, 3 aging) at N=64.

For each video, loads the saved segmentation results from data/segmented/ and
reconstructs the exact preview that KneeSegmentationPipeline.display_saved_results()
would show interactively. Saves each preview as both MP4 video and TIF stack.

Outputs are saved to: figures/segmentation_previews/
Naming: {condition}_{video_id}_N64_preview.(mp4|tif)

Usage:
    python scripts/visualization/export_saved_segmentation_previews.py
"""

import numpy as np
from pathlib import Path
from utils import io, views
import tifffile as tif
from config.knee_metadata import get_knee_meta

# Target videos: all 7 at N=64
# Map (condition, video_id) to the actual file naming convention used in data/segmented/
TARGETS = [
    ("normal", "0308"),  # 308 -> 0308 (zero-padded)
    ("normal", "1190"),
    ("normal", "1193"),
    ("normal", "1207"),
    ("aging", "1339"),
    ("aging", "1342"),
    ("aging", "1358"),
]

N_SEGMENTS = 64
OUTPUT_DIR = Path("figures/segmentation_previews")

def export_preview_for_video(condition: str, video_id: str, n_segments: int) -> None:
    """
    Export MP4 and TIF previews for a single video's saved segmentation results.
    """
    print(f"\n=== Processing {condition}_{video_id} (N={n_segments}) ===")

    # Build file paths (matching KneeSegmentationPipeline.save_results() convention)
    base_name = f"{condition}_{video_id}_"
    video_path = Path("data/segmented") / f"{base_name}video_N{n_segments}.npy"
    radial_path = Path("data/segmented") / f"{base_name}radial_N{n_segments}.npy"
    femur_path = Path("data/segmented") / f"{base_name}femur_N{n_segments}.npy"

    # Verify files exist
    for path in [video_path, radial_path, femur_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing saved result file: {path}")

    # Load saved arrays
    print(f"Loading saved results from {Path('data/segmented')}")
    video = io.load_nparray(video_path)
    radial = io.load_nparray(radial_path)
    femur = io.load_nparray(femur_path)  # Not used in preview, but verify it exists

    print(f"Video shape: {video.shape}, dtype: {video.dtype}")
    print(f"Radial mask shape: {radial.shape}, dtype: {radial.dtype}")

    # Reconstruct the preview exactly as _show_radial_preview() does
    print("Building preview...")

    # 1. Radial mask grayscale visualization (left side of concat)
    radial_gray = radial * (255 // n_segments)
    radial_gray = radial_gray.astype(np.uint8)

    # 2. Boundary overlay on processed video (right side of concat)
    # Start with processed video for boundary overlays
    boundary_overlay = video.copy()

    # Reference boundary between seg 1 and seg N (brightest)
    boundary_overlay = views.draw_boundary_line(
        boundary_overlay,
        radial,
        seg_num=1,
        n_segments=n_segments,
        intensity=255,  # brightest
        thickness=1,
        show_video=False,
    )

    # Anatomical region boundaries if metadata available
    try:
        meta = get_knee_meta(condition, int(video_id), n_segments)
        regions = meta.regions
        jc = regions.get("JC")
        ot = regions.get("OT")
        sb = regions.get("SB")
        if jc and ot and sb:
            # Boundary between JC and OT (mid-bright)
            boundary_overlay = views.draw_boundary_line(
                boundary_overlay,
                radial,
                seg_num=ot.start,  # boundary between JC and OT
                n_segments=n_segments,
                intensity=200,  # slightly dimmer
                thickness=1,
                show_video=False,
            )

            # Boundary between OT and SB (dimmer)
            boundary_overlay = views.draw_boundary_line(
                boundary_overlay,
                radial,
                seg_num=sb.start,  # boundary between OT and SB
                n_segments=n_segments,
                intensity=150,  # dimmer still
                thickness=1,
                show_video=False,
            )
    except Exception as e:
        print(f"Warning: no region metadata for {condition}_{video_id}_N{n_segments}, skipping anatomical boundaries: {e}")

    # Outer radial mask boundary
    boundary_overlay = views.draw_outer_radial_mask_boundary(
        boundary_overlay, radial
    )

    # 3. Horizontally concatenate (exactly like views.show_frames([radial_gray, boundary_overlay]))
    # Both should have shape (n_frames, height, width)
    assert radial_gray.shape == boundary_overlay.shape, f"Shape mismatch: {radial_gray.shape} vs {boundary_overlay.shape}"

    # Stack horizontally: [radial_gray | boundary_overlay]
    preview = np.concatenate([radial_gray, boundary_overlay], axis=2)

    print(f"Preview shape: {preview.shape}, dtype: {preview.dtype}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save MP4
    mp4_path = OUTPUT_DIR / f"{condition}_{video_id}_N{n_segments}_preview.mp4"
    print(f"Saving MP4 to {mp4_path}")
    io.save_mp4(str(mp4_path), preview, fps=30)

    # Save TIF stack
    tif_path = OUTPUT_DIR / f"{condition}_{video_id}_N{n_segments}_preview.tif"
    print(f"Saving TIF stack to {tif_path}")
    tif.imwrite(str(tif_path), preview.astype(np.uint8))

    print(f"✅ Exported {condition}_{video_id} previews")


def main():
    """Export previews for all target videos."""
    print("Starting export of segmentation previews for 7 knee videos...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Targets: {TARGETS}")

    for condition, video_id in TARGETS:
        try:
            export_preview_for_video(condition, video_id, N_SEGMENTS)
        except Exception as e:
            print(f"❌ Failed to export {condition}_{video_id}: {e}")
            raise

    print("\n🎉 All previews exported successfully!")
    print(f"Check {OUTPUT_DIR} for 7 MP4 videos and 7 TIF stacks.")


if __name__ == "__main__":
    main()
