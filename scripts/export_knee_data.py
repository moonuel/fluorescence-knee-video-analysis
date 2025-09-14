"""Template script for saving comprehensive data for knee video analysis in a consistent format into an Excel spreadsheet file.

Data format:
    Sheet 1: Total Pixel Intensity per radial segment, for every frame
    Sheet 2: Total number of non-zero pixels per radial segment, for every frame
    Sheet 3: File number, total number of frames, frame numbers for each cycle, and segment numbers assigned to the left/middle/right parts of the knee. 
"""

from utils import utils, io, views
import numpy as np
import pandas as pd
import pandas as pd
from pathlib import Path


def load_masks(filepath:str) -> np.ndarray:
    """Loads the mask at the specified location. 
    Handles both old uint8 radial masks with shape (nmasks, nframes, h, w) and new uint8 radial masks with shape (nframes, h, w).
    
    Old mask arrays are very space-inefficient and have one dimension for each segment. New mask arrays use a unique numerical label from {1...N} instead."""

    masks = io.load_nparray(filepath)

    if not masks.dtype == np.uint8:
        raise ValueError(f"File is not of type uint8. Is it a radial mask? Given: {masks.dtype=}")
    
    # Convert inefficient mask array to efficient array
    if len(masks.shape) == 4:
        N = masks.shape[0] # Expected shape: (nmasks, nframes, h, w)
        masks_bool = np.zeros(shape=masks.shape[1:], dtype=np.uint8) # Expected shape: (nframes, h, w)
        for n in range(N):
            masks_bool[masks[n] > 0] = n+1 # Convert each slice of inefficient array to a numerical label from {1...N}
        masks = masks_bool

    assert len(masks.shape) == 3 # Soft check that output is shape (nfs, h, w)

    return masks


def load_video(filepath:str) -> np.ndarray:
    """Loads the video at the specified location."""
    video = io.load_nparray(filepath)

    if not video.dtype == np.uint8: 
        raise ValueError(f"File is not of type uint8. Is it a video? Given: {video.dtype=}")
    
    if not len(video.shape) == 3:
        raise ValueError(f"File is not compatible with shape (nfs, h, w). Is it a grayscale video? Given: {video.shape=}")
    
    return video


def build_metadata_row(metadata: dict) -> dict:
    """
    Convert metadata dict (with nested cycles) into a flat dict suitable for a DataFrame row.
    """
    row = {}

    for key, value in metadata.items():
        if key != "cycle_frames":
            if isinstance(value, list):
                row[key] = ", ".join(map(str, value))
            else:
                row[key] = value

    for cycle_name, cvals in metadata.get("cycle_frames", {}).items():
        row[f"{cycle_name} flexion"] = f"{cvals['flexion'][0]}:{cvals['flexion'][1]}"
        row[f"{cycle_name} extension"] = f"{cvals['extension'][0]}:{cvals['extension'][1]}"

    return row


def save_analysis_to_excel(total_sums: np.ndarray,
                           total_nonzero: np.ndarray,
                           metadata: dict,
                           output_file: str | Path):
    """
    Save knee analysis results to an Excel file with three sheets:
      - Total sums (N x nframes)
      - Total nonzero (N x nframes)
      - Metadata (one row)
    """
    output_file = Path(output_file)

    # Metadata DataFrame
    df_meta = pd.DataFrame([build_metadata_row(metadata)])

    # Convert arrays to DataFrames
    N, nframes = total_sums.shape[0], total_sums.shape[1]

    df_sums = pd.DataFrame(total_sums.T, columns=[f"Segment {i}" for i in range(1, N + 1)])
    df_sums.index.name = "Frame"

    df_nonzero = pd.DataFrame(total_nonzero.T, columns=[f"Segment {i}" for i in range(1, N + 1)])
    df_nonzero.index.name = "Frame"

    # Write all three sheets
    with pd.ExcelWriter(output_file) as writer:
        df_sums.to_excel(writer, sheet_name="Sum of Pixel Intensities (0-255)")
        df_nonzero.to_excel(writer, sheet_name="Number of Non-zero Pixels (Size of Mask)")
        df_meta.to_excel(writer, sheet_name="Analysis Metadata", index=False)

    print(f"✅ Analysis results saved to {output_file.resolve()}")


def main(mask_path, video_path):

    # Load data 
    shared_dir = "../data/processed/"
    mask_path =  shared_dir + mask_path # Manually specify filenames 
    video_path = shared_dir + video_path

    masks = load_masks(mask_path)
    video = load_video(video_path)
    
    assert masks.shape == video.shape # Sanity check that we're using new mask format
    nfs, h, w = video.shape
    print(f"{video.shape=}")

    mask_lbls = np.unique(masks[masks > 0]).astype(int) # Returns sorted list of unique non-zero labels
    N = len(mask_lbls)

    # Validate data
    print(f"{mask_lbls=}")
    views.show_frames(masks * (255 // mask_lbls.max())) # Rescale label intensities for better viewing
    views.show_frames(video)

    # Calculate total pixel intensities within each segment of the video
    total_sums = np.zeros(shape=(N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_sums[n, f] = frame[mask_f == lbl].sum()

    # Calculate number of non-zero pixels within each segment of the video (for normalization purposes)
    total_nonzero = np.zeros((N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_nonzero[n, f] = np.count_nonzero(frame[mask_f == lbl])

    assert total_sums.shape == total_nonzero.shape # Sanity check

    print(f"{total_sums[:, 0]=}")
    print(f"{total_nonzero[:, 0]=}")

    return total_sums, total_nonzero


# Example usage:

#     mask_path = ".npy"
#     video_path = ".npy"
    
#     total_sums, total_nonzero = main(mask_path, video_path)

#     metadata = {
#         "Description": "Contains metadata about the knee analysis data. "
#             "All frame indices are given in 0-based indexing format, "
#             "i.e. video[0] denotes the first frame (as opposed to video[1] denoting the first frame). ",
#         "file_number": "",
#         "type_of_knee": "",
#         "frame_range": "",
#         "cycle_frames": {
#             "Cycle 1": {"flexion": (), "extension": ()}, # Tuple[int, int]
#         },
#         "num_of_segments": "16",
#         "left_segments": "",
#         "middle_segments": "",
#         "right_segments": ""
#     }

#     save_analysis_to_excel(total_sums, total_nonzero, metadata, "_analysis_data.xlsx")

if __name__ == "__main__":
    
    mask_name = "XXXX_knee_radial_masks_N16.npy" # Path will be pre-pended
    video_name = "XXXX_knee_radial_video_N16.npy"
    
    total_sums, total_nonzero = main(mask_name, video_name)
    
    metadata = {
        "Description": "Contains metadata about the knee analysis data. "
            "All frame indices are given in 0-based indexing format, "
            "i.e. video[0] denotes the first frame (as opposed to video[1] denoting the first frame). ",
        "file_number": "XXXX",
        "type_of_knee": "normal", # "aging" or "normal"
        "frame_range": "XXX:XXX",
        "cycle_frames": {
            "Cycle 1": {"flexion": (), "extension": ()}, # Tuple[int, int]
        },
        "num_of_segments": "16",
        "left_segments": "",
        "middle_segments": "",
        "right_segments": ""
    }

    save_analysis_to_excel(total_sums, total_nonzero, metadata, "XXXX_analysis_data.xlsx")









"""aging 1339 data"""

    # mask_path = "1339_knee_radial_masks_N16.npy"
    # video_path = "1339_knee_radial_video_N16.npy"
    
    # total_sums, total_nonzero = main(mask_path, video_path)

    # metadata = {
    #     "Description": "Contains metadata about the knee analysis data. "
    #         "All frame indices are given in 0-based indexing format, "
    #         "i.e. video[0] denotes the first frame (as opposed to video[1] denoting the first frame). ",
    #     "file_number": "1339",
    #     "type_of_knee": "aging",
    #     "frame_range": "289:608",
    #     "cycle_frames": {
    #         "Cycle 1": {"flexion": (290, 309), "extension": (312, 329)},
    #         "Cycle 2": {"flexion": (331, 352), "extension": (355, 374)},
    #         "Cycle 3": {"flexion": (375, 394), "extension": (398, 421)},
    #         "Cycle 4": {"flexion": (422, 439), "extension": (441, 463)},
    #         "Cycle 5": {"flexion": (464, 488), "extension": (490, 512)},
    #         "Cycle 6": {"flexion": (513, 530), "extension": (532, 553)},
    #         "Cycle 7": {"flexion": (554, 576), "extension": (579, 609)}
    #     },
    #     "num_of_segments": "16",
    #     "left_segments": "11, 12, 13, 14, 15, 16, 1",
    #     "middle_segments": "7, 8, 9, 10",
    #     "right_segments": "1, 2, 3, 4, 5, 6"
    # }

    # save_analysis_to_excel(total_sums, total_nonzero, metadata, "1339_analysis_data.xlsx")