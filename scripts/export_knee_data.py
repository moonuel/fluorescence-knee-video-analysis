"""Template script for saving comprehensive data for knee video analysis in a consistent format into an Excel spreadsheet file.

Data format:
    Sheet 1: Total Pixel Intensity per radial segment, for every frame
    Sheet 2: Total number of non-zero pixels per radial segment, for every frame
    Sheet 3: File number, total number of frames, frame numbers for each cycle, and segment numbers assigned to the left/middle/right parts of the knee. 
"""

from utils import utils, io, views
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Tuple


@dataclass
class Cycle:
    flexion: Tuple[int, int]
    extension: Tuple[int, int]

    def validate(self) -> None:
        for name, rng in {"flexion": self.flexion, "extension": self.extension}.items():
            if rng[0] > rng[1]:
                raise ValueError(f"{name} range {rng} is invalid: start must be <= end")


@dataclass
class Metadata:
    description: str = (
        "Contains metadata about the knee analysis data. "
        "All frame indices are given in 0-based indexing format, "
        "i.e. video[0] denotes the first frame (as opposed to video[1] denoting the first frame). "
        "All frame ranges are given in inclusive format, "
        "i.e. 'cycle1.flexion = (100, 200)' means that 200 is the last frame in the cycle, "
        "for 101 total frames."
    )
    file_number: int = 0
    type_of_knee: str = ""
    frame_start: int = 0
    frame_end: int = 0
    total_frames: int = field(init=False)
    cycle_frames: List[Cycle] = field(default_factory=list)  # ✅ now an ordered list
    num_of_segments: int = 0
    segment_labels: List[int] = field(default_factory=list)
    left_segments: List[int] = field(default_factory=list)
    middle_segments: List[int] = field(default_factory=list)
    right_segments: List[int] = field(default_factory=list)

    def __post_init__(self):
        # Automatically compute total_frames
        self.total_frames = self.frame_end - self.frame_start + 1

    def validate(self) -> None:
        # frame range
        if self.frame_start >= self.frame_end:
            raise ValueError(
                f"frame_start={self.frame_start} must be less than frame_end={self.frame_end}"
            )
        
        print(f"frame_start={self.frame_start}, frame_end={self.frame_end}")
        print(f"total_frames={self.total_frames}")

        # validate cycles
        for i, cycle in enumerate(self.cycle_frames):
            cycle.validate()
            for rng in (cycle.flexion, cycle.extension):
                if not (self.frame_start <= rng[0] and rng[1] <= self.frame_end):
                    raise ValueError(
                        f"Cycle {i} {rng} is outside frame range "
                        f"[{self.frame_start}, {self.frame_end}]"
                    )

        print("all cycles in the frame range")

        # check number of segments matches
        all_segments = set(self.left_segments + self.middle_segments + self.right_segments)
        if len(all_segments) != self.num_of_segments:
            raise ValueError(
                f"num_of_segments={self.num_of_segments} does not match "
                f"unique segments provided={len(all_segments)}"
            )

        print("number of segments matches")

        # check segment_labels matches declared segments
        if sorted(all_segments) != sorted(self.segment_labels):
            raise ValueError(
                f"segment_labels {self.segment_labels} do not match segments declared in "
                f"left/middle/right {sorted(all_segments)}"
            )
        
        print("all segments accounted for ")


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
                           metadata: "Metadata",
                           output_file: str | Path):
    """
    Save knee analysis results to an Excel file with three sheets:
      - Total sums (N x nframes)
      - Total nonzero (N x nframes)
      - Metadata (one row)
    """
    output_file = Path(output_file)

    # --- Convert Metadata dataclass into a dict row ---
    meta_dict = asdict(metadata)

    df_meta = pd.DataFrame([meta_dict])

    # --- Convert arrays to DataFrames ---
    N, nframes = total_sums.shape[0], total_sums.shape[1]

    # Use the true frame numbers from metadata
    frame_index = range(metadata.frame_start, metadata.frame_end + 1)

    df_sums = pd.DataFrame(
        total_sums.T,
        index=frame_index,
        columns=[f"Segment {i}" for i in range(1, N + 1)]
    )
    df_sums.index.name = "Frame"

    df_nonzero = pd.DataFrame(
        total_nonzero.T,
        index=frame_index,
        columns=[f"Segment {i}" for i in range(1, N + 1)]
    )
    df_nonzero.index.name = "Frame"

    # --- Write all three sheets ---
    with pd.ExcelWriter(output_file) as writer:
        df_sums.to_excel(writer, sheet_name="Sum of Pixel Intensities (0-255)")
        df_nonzero.to_excel(writer, sheet_name="Number of Non-zero Pixels (Size of Mask)")
        df_meta.to_excel(writer, sheet_name="Analysis Metadata", index=False)

    print(f"✅ Analysis results saved to {output_file.resolve()}")


def compute_sums_nonzeros(mask_path, video_path):

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

# Store cycle ranges here
CYCLES = {
    1207: "242-254	264-280	281-293	299-312	318-335	337-352	353-372	373-389	391-411	412-431	434-451	453-467	472-486	488-505	614-632	633-651	652-671	672-690	693-708	709-727	731-748	751-767	768-786	787-804	807-822	824-841	844-862	863-877",
    1190: "66-89	92-109 421-452	470-492	503-532	533-569 737-767	770-793	794-822	823-860",
    1193: "1793-1802 1803-1813 1814-1823 1824-1834 1835-1844 1845-1853 1854-1864 1865-1873 1874-1882 1882-1890"
}

def main():
    
    # Select data
    video_id = 1193
    type = "normal"
    N = 64
    
    print(f"{video_id=}, {type=}, {N=}")
    breakpoint()
    # -------------------------------------------------------------------------------
    masks = load_masks(f"../data/processed/{video_id}_{type}_radial_masks_N{N}.npy")
    video = load_video(f"../data/processed/{video_id}_{type}_radial_video_N{N}.npy")
    cycles = [c.split("-") for c in CYCLES[video_id].split()]

    # Compute within-segment total intensities, and number of pixels in each segment
    nfs = video.shape[0]
    mask_lbls = np.unique(masks[masks > 0]).astype(int) # Returns sorted list of unique non-zero labels

    total_sums = np.zeros(shape=(N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_sums[n, f] = frame[mask_f == lbl].sum()

    total_nonzero = np.zeros((N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_nonzero[n, f] = np.count_nonzero(frame[mask_f == lbl])

    # Write intensity data and cycle data into Excel spreadsheet
    total_sums = pd.DataFrame(total_sums.T)
    total_nonzero = pd.DataFrame(total_nonzero.T)
    flex = pd.DataFrame(cycles[::2])
    ext = pd.DataFrame(cycles[1::2])

    total_sums.index = total_sums.index + 1; total_nonzero.index = total_nonzero.index + 1 # 1-indexing
    flex.index = flex.index + 1; ext.index = ext.index + 1

    total_sums.columns = total_sums.columns + 1; total_nonzero.columns = total_nonzero.columns + 1 # Formatting
    flex.columns = ["Start", "End"]; ext.columns = ["Start", "End"]

    print("-----------------------------------------------------------")
    print(total_sums)
    print(flex)
    print(ext)

    breakpoint()

    with pd.ExcelWriter(f"../data/video_intensities/video{video_id}N{N}.xlsx") as writer:
        total_sums.to_excel(writer, sheet_name="Segment Intensities", index=True)
        flex.to_excel(writer, sheet_name="Flexion Frames", index=True)
        ext.to_excel(writer, sheet_name="Extension Frames", index=True)
        # total_nonzero.to_excel(writer, sheet_name="Number of Mask Pixels", index=True)



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

# def main():
    
#     mask_name = "1339_knee_radial_masks_N16.npy" # Path will be pre-pended
#     video_name = "1339_knee_radial_video_N16.npy"
    
#     total_sums, total_nonzero = compute_sums_nonzeros(mask_name, video_name)
    
#     # Create cycle objects
#     c1 = Cycle(flexion=(289, 308), extension=(311, 328))
#     c2 = Cycle(flexion=(330, 351), extension=(354, 373))
#     c3 = Cycle(flexion=(374, 393), extension=(397, 420))
#     c4 = Cycle(flexion=(421, 438), extension=(440, 462))
#     c5 = Cycle(flexion=(463, 487), extension=(489, 511))
#     c6 = Cycle(flexion=(512, 529), extension=(531, 552))
#     c7 = Cycle(flexion=(553, 575), extension=(578, 608))

#     # Create metadata object
#     metadata = Metadata(
#         file_number=1339,
#         type_of_knee="aging",
#         frame_start=289,
#         frame_end=608,
#         cycle_frames=[c1, c2, c3, c4, c5, c6, c7],
#         num_of_segments=16,
#         segment_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
#         left_segments=[11,12,13,14,15,16,1],
#         middle_segments=[7,8,9,10],
#         right_segments=[1,2,3,4,5,6],
#     )

#     # Validate consistency
#     metadata.validate()

#     print(metadata.total_frames)  # 320

#     save_analysis_to_excel(total_sums, total_nonzero, metadata, "1339_analysis_data.xlsx")





if __name__ == "__main__":
    main()



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
    #     "left_segments": "11, 12, 13, 14, 15, 16",
    #     "middle_segments": "7, 8, 9, 10",
    #     "right_segments": "1, 2, 3, 4, 5, 6"
    # }

    # save_analysis_to_excel(total_sums, total_nonzero, metadata, "1339_analysis_data.xlsx")