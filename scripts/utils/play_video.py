from utils import io, views
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import argparse
import cv2
import sys
import re

from config.knee_metadata import get_knee_meta_by_condition

# Get project root directory for robust path handling
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "video_previews"


def discover_available_videos(segmented_dir: Path) -> Dict[int, Dict]:
    """
    Scan directory for segmented video files and extract metadata.
    Returns: {video_id: {'type': str, 'N_values': set}}
    """
    videos = {}
    pattern = "*_video_N*.npy"

    for file in segmented_dir.glob(pattern):
        # Parse: "aging_1339_video_N64.npy" or "normal_0308_video_N16.npy"
        parts = file.stem.split("_")
        knee_type = parts[0]  # "aging" or "normal"
        video_id_str = parts[1]  # "1339" or "0308"
        N_part = parts[3]  # "N64" or "N16"

        video_id = int(video_id_str)
        N = int(N_part[1:])  # Extract number after 'N'

        if video_id not in videos:
            videos[video_id] = {'type': knee_type, 'N_values': set()}
        videos[video_id]['N_values'].add(N)

    return videos


def parse_video_selector(text: str) -> Tuple[int, Optional[int]]:
    """Parse selector into (video_id, N).

    Accepted:
      - "1339" -> (1339, None) raw/unsegmented
      - "1339N64" / "1339n64" -> (1339, 64) segmented
    """
    s = text.strip()
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1)), None

    m = re.fullmatch(r"(\d+)[Nn](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(
        "Invalid selector. Use <video_id> for raw or <video_id>N<N> for segmented (e.g. 1339N64)."
    )


def discover_raw_videos(raw_dir: Path) -> Dict[int, Dict]:
    """Discover raw/unsegmented previewable files in data/raw.

    Returns {video_id: {'files': list[Path]}}.

    Only supports .npy and .avi (non-recursive scan).
    Video id is extracted from trailing "_<digits>" in the stem.
    """
    videos: Dict[int, Dict] = {}
    if not raw_dir.exists():
        return videos

    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".npy", ".avi"}:
            continue

        m = re.search(r"_(\d+)$", p.stem)
        if not m:
            continue
        video_id = int(m.group(1))
        videos.setdefault(video_id, {"files": []})["files"].append(p)

    for info in videos.values():
        info["files"].sort(key=lambda x: (0 if x.suffix.lower() == ".npy" else 1, x.name.lower()))
    return videos


def choose_preferred_raw_file(files: list[Path]) -> Path:
    if not files:
        raise FileNotFoundError("No raw files available")
    files_sorted = sorted(files, key=lambda x: (0 if x.suffix.lower() == ".npy" else 1, x.name.lower()))
    return files_sorted[0]


def get_region_boundary_segments(knee_type: str, video_id: int, N: int) -> Optional[Tuple[int, int]]:
    """Get JC/OT and OT/SB boundary segment numbers from knee metadata."""
    try:
        meta = get_knee_meta_by_condition(knee_type, int(video_id), int(N))
    except Exception as e:
        print(f"Warning: no region metadata for {knee_type}_{video_id}_N{N}: {e}")
        return None

    regions = meta.regions
    jc = regions.get("JC")
    ot = regions.get("OT")
    sb = regions.get("SB")
    if not (jc and ot and sb):
        print(f"Warning: incomplete region metadata for {knee_type}_{video_id}_N{N}")
        return None

    return ot.s, sb.s


def ensure_uint8(video: np.ndarray) -> np.ndarray:
    """Normalize preview data to uint8 for display and save parity."""
    if video.dtype == np.uint8:
        return video
    if np.issubdtype(video.dtype, np.floating):
        return np.clip(video * 255.0, 0, 255).astype(np.uint8)
    return np.clip(video, 0, 255).astype(np.uint8)


def stack_preview_panels(*panels: np.ndarray) -> np.ndarray:
    """Horizontally concatenate one or more grayscale preview panels."""
    if not panels:
        raise ValueError("At least one preview panel is required")
    if len(panels) == 1:
        return ensure_uint8(panels[0])

    base_shape = panels[0].shape
    if any(panel.shape != base_shape for panel in panels[1:]):
        raise ValueError("All preview panels must have the same shape")

    return ensure_uint8(np.concatenate([ensure_uint8(panel) for panel in panels], axis=2))


def burn_frame_numbers(video: np.ndarray, frame_offset: int = 0) -> np.ndarray:
    """Render frame numbers using the same overlay style as views.show_frames()."""
    rendered = ensure_uint8(video).copy()
    if rendered.ndim == 2:
        rendered = rendered.reshape(1, *rendered.shape)

    _, h, _ = rendered.shape
    btm_l_pos = (10, h - 10)

    for frame_num in range(rendered.shape[0]):
        frame = rendered[frame_num]
        cv2.rectangle(frame, (0, h - 32), (75, h), color=0, thickness=-1)
        cv2.putText(
            frame,
            str(frame_num + frame_offset),
            btm_l_pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=255,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return rendered


def build_output_path(
    video_id: int,
    N: Optional[int],
    knee_type: Optional[str],
    raw_path: Optional[Path],
    segmented_only: bool,
) -> Path:
    """Construct the default MP4 output path for the requested preview."""
    if N is not None:
        assert knee_type is not None
        mode = "video-only" if segmented_only else "composite"
        filename = f"segmentation_preview_{knee_type}_{video_id:04d}_N{N}_mode-{mode}.mp4"
    else:
        assert raw_path is not None
        filename = f"raw_preview_{video_id:04d}_source-{raw_path.stem}.mp4"

    return DEFAULT_OUT_DIR / filename


def parse_arguments():
    """
    Parse command line arguments with dynamic validation based on available files.
    """
    parser = argparse.ArgumentParser(
        description="Play knee videos (segmented or raw)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python play_video.py --list\n"
               "  python play_video.py --list-raw\n"
               "  python play_video.py 1339N64\n"
               "  python play_video.py 1339N64 -S\n"
               "  python play_video.py 1339N64 --save\n"
               "  python play_video.py 1339N64 -S --save\n"
               "  python play_video.py 1339\n"
               "  python play_video.py 1339 64  # legacy"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available video IDs and segment counts"
    )

    parser.add_argument(
        '--list-raw',
        action='store_true',
        help="List raw/unsegmented previewable videos in data/raw"
    )

    parser.add_argument(
        'selector',
        nargs='?',
        help="Video selector: <id> (raw) or <id>N<N> (segmented), e.g. 1339 or 1339N64"
    )

    parser.add_argument(
        'N',
        type=int,
        nargs='?',
        help="Legacy segmented mode: play_video.py <id> <N>"
    )

    parser.add_argument(
        '-S', '--segmented-only',
        action='store_true',
        help="For segmented inputs, show/save only the processed segmented video panel"
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help="Save the rendered preview as an MP4 under figures/video_previews/"
    )

    args = parser.parse_args()

    # Discover available videos
    segmented_dir = PROJECT_ROOT / "data" / "segmented"
    available = discover_available_videos(segmented_dir)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_available = discover_raw_videos(raw_dir)

    # Handle --list mode
    if args.list:
        print("\nAvailable videos in data/segmented:\n")
        for vid, info in sorted(available.items()):
            n_vals = ', '.join(f"N={n}" for n in sorted(info['N_values']))
            print(f"  {vid:4d} ({info['type']:6s}) - {n_vals}")
        sys.exit(0)

    if args.list_raw:
        print("\nAvailable videos in data/raw (previewable: .npy, .avi):\n")
        for vid, info in sorted(raw_available.items()):
            files = ", ".join(p.name for p in info["files"])
            print(f"  {vid:4d} - {files}")
        sys.exit(0)

    # Validate required arguments
    if args.selector is None:
        parser.error("selector is required (or use --list/--list-raw)")

    # Backward-compat: play_video.py <id> <N>
    selector_text = args.selector
    if args.N is not None and re.fullmatch(r"\d+", selector_text.strip()):
        selector_text = f"{selector_text}N{args.N}"

    try:
        video_id, N = parse_video_selector(selector_text)
    except ValueError as e:
        parser.error(str(e))

    if N is not None:
        # Segmented validation
        if video_id not in available:
            valid_ids = ', '.join(str(v) for v in sorted(available.keys()))
            parser.error(
                f"Video ID {video_id} not found in data/segmented.\n"
                f"Available IDs: {valid_ids}\n"
                f"Use --list to see details."
            )
        if N not in available[video_id]['N_values']:
            valid_n = ', '.join(f"N={n}" for n in sorted(available[video_id]['N_values']))
            parser.error(
                f"N={N} not available for video {video_id}.\n"
                f"Available: {valid_n}"
            )
        return video_id, N, available[video_id]['type'], None, args.segmented_only, args.save

    # Raw validation
    if video_id not in raw_available:
        valid_ids = ', '.join(str(v) for v in sorted(raw_available.keys()))
        parser.error(
            f"Video ID {video_id} not found in data/raw.\n"
            f"Available IDs: {valid_ids}\n"
            f"Use --list-raw to see details."
        )
    raw_path = choose_preferred_raw_file(raw_available[video_id]["files"])
    return video_id, None, None, raw_path, args.segmented_only, args.save


def main(
    video_id: int,
    N: Optional[int],
    knee_type: Optional[str],
    raw_path: Optional[Path],
    segmented_only: bool = False,
    save: bool = False,
):
    """
    Display the radial segmentation mask alongside the processed video.
    """
    preview_video: np.ndarray
    preview_title: str

    if N is not None:
        segmented_dir = PROJECT_ROOT / "data" / "segmented"
        assert knee_type is not None

        radial_path = segmented_dir / f"{knee_type}_{video_id:04d}_radial_N{N}.npy"
        video_path = segmented_dir / f"{knee_type}_{video_id:04d}_video_N{N}.npy"
        femur_path = segmented_dir / f"{knee_type}_{video_id:04d}_femur_N{N}.npy"
        if not radial_path.exists():
            raise FileNotFoundError(f"Missing radial mask: {radial_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")
        if not femur_path.exists():
            raise FileNotFoundError(f"Missing femur mask: {femur_path}")

        print(f"Loading radial mask: {radial_path}")
        radial_mask = io.load_nparray(radial_path)

        print(f"Loading video: {video_path}")
        video = io.load_nparray(video_path)

        print(f"Loading femur mask: {femur_path}")
        femur_mask = io.load_nparray(femur_path)

        max_label = np.max(radial_mask)
        if max_label > 0:
            scale = 255 // max_label
            radial_display = radial_mask * scale
        else:
            radial_display = radial_mask

        radial_display = ensure_uint8(radial_display)

        print(f"Displaying {knee_type} video {video_id} with {N} segments...")
        if segmented_only:
            print("Processed segmented video only")
        else:
            print("Femur mask (left), radial mask (middle), and boundary overlay (right)")

        boundary_overlay = video.copy()

        try:
            boundary_overlay = views.draw_boundary_line(
                boundary_overlay,
                radial_mask,
                seg_num=1,
                n_segments=N,
                intensity=255,
                thickness=1,
                show_video=False,
            )
        except Exception as e:
            print(f"Warning: failed to draw femur line (seg 1/N boundary) ({e})")

        region_bounds = get_region_boundary_segments(knee_type, int(video_id), int(N))
        if region_bounds is not None:
            jc_ot_seg, ot_sb_seg = region_bounds

            try:
                boundary_overlay = views.draw_boundary_line(
                    boundary_overlay,
                    radial_mask,
                    seg_num=jc_ot_seg,
                    n_segments=N,
                    intensity=200,
                    thickness=1,
                    show_video=False,
                )
                boundary_overlay = views.draw_boundary_line(
                    boundary_overlay,
                    radial_mask,
                    seg_num=ot_sb_seg,
                    n_segments=N,
                    intensity=150,
                    thickness=1,
                    show_video=False,
                )
            except Exception as e:
                print(f"Warning: failed to draw anatomical boundaries ({e})")

        try:
            boundary_overlay = views.draw_outer_radial_mask_boundary(
                boundary_overlay,
                radial_mask,
                intensity=255,
                thickness=1,
                show_video=False,
            )
        except Exception as e:
            print(f"Warning: failed to draw outer segmentation boundary ({e})")

        preview_title = f"Radial Segmentation Preview: {knee_type}_{video_id:04d}_N{N}"
        if segmented_only:
            preview_video = ensure_uint8(boundary_overlay)
        else:
            preview_video = stack_preview_panels(femur_mask, radial_display, boundary_overlay)
    else:
        assert raw_path is not None
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw file: {raw_path}")

        print(f"Loading raw video: {raw_path}")
        if raw_path.suffix.lower() == ".npy":
            video = io.load_nparray(raw_path)
        elif raw_path.suffix.lower() == ".avi":
            video = io.load_avi(str(raw_path))
        else:
            raise ValueError(f"Unsupported raw extension: {raw_path.suffix}")

        print(f"Displaying raw video {video_id} from {raw_path.name}...")
        preview_title = f"raw_{video_id:04d}"
        preview_video = ensure_uint8(video)

    rendered_preview = burn_frame_numbers(preview_video)

    if save:
        out_path = build_output_path(video_id, N, knee_type, raw_path, segmented_only)
        print(f"Saving preview to {out_path}")
        io.save_mp4(str(out_path), rendered_preview, fps=30)

    views.show_frames(rendered_preview, title=preview_title, show_num=False)


if __name__ == "__main__":
    video_id, N, knee_type, raw_path, segmented_only, save = parse_arguments()
    main(video_id, N, knee_type, raw_path, segmented_only=segmented_only, save=save)
