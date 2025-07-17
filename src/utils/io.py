import pandas as pd
import numpy as np
import os
from tifffile import imread as tif_imread
from config import VERBOSE
from typing import Tuple, Dict, Union
import cv2
from pathlib import Path
import cv2
import h5py
from typing import Optional


def load_aging_knee_coords(filename:str, knee_id:Union[str, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Inputs:
        filename (str) - path to the .xlsx coordinates file to be loaded
        knee_id (str|int) - name or index of the Excel sheet to be used 

    Outputs:
        coords (pd.DataFrame) - contains the pairs of coordinates provided by Huizhu @ Fudan University
        metadata (Dict) - contains information mostly relevant for plotting. See keys for info.
    """
    if VERBOSE: print("load_aging_knee_coords() called!")

    # Implement function overload with knee_id as int
    if isinstance(knee_id, int): 
        if VERBOSE: print(" > overloaded func!")
        with pd.ExcelFile(filename, engine="openpyxl") as xl:
            sheet_names = xl.sheet_names
        knee_id = sheet_names[knee_id]
    
    # Import knee coordinates
    coords_sheet = pd.read_excel(filename, engine='openpyxl', sheet_name=knee_id) # More updated Excel import
    # coords_sheet = coords_sheet[knee_id]

    # Clean data
    coords_sheet.drop(columns=['Unnamed: 0', 'Unnamed: 5'], axis=1, inplace=True)

    na_coords_1 = coords_sheet['Frame Number'].isna() & coords_sheet['X'].isna() & coords_sheet['Y'].isna() # What was I cooking
    na_coords_2 = coords_sheet['Frame Number.1'].isna() & coords_sheet['X.1'].isna() & coords_sheet['Y.1'].isna()

    coords_1 = coords_sheet[['Frame Number', 'X', 'Y']].loc[~na_coords_1]
    coords_2 = coords_sheet[['Frame Number.1', 'X.1', 'Y.1']].loc[~na_coords_2]

    # Record metadata
    flx_ext_pt = int(coords_2.iloc[0]['Frame Number.1']) # flexion/extension boundary for plotting

    # Reformat data
    coords_2.rename(columns={'Frame Number.1': 'Frame Number', 'X.1': 'X', 'Y.1': 'Y'}, inplace=True) 
    coords = pd.concat([coords_1, coords_2], axis=0)

    # Set frame number as index
    coords.set_index("Frame Number", inplace=True)
    coords.index = coords.index.to_series().fillna(method="ffill").astype(int)
    uqf = coords.index.unique()

    assert coords.shape[1] == 2 # only want cols X,Y

    metadata = {"knee_id": knee_id, "flx_ext_pt": flx_ext_pt, "f0": uqf[0], "fN": uqf[-1]}
    return coords, metadata

def load_normal_knee_coords(fn:str, sheet_num:int) -> pd.DataFrame:
    if VERBOSE: print("load_normal_knee_coords() called!")

    sheet_names=["8.29 re-measure", "8.29 2nd", "8.29 3rd", "8.6"]
    coords = pd.read_excel(fn, engine="openpyxl", sheet_name=sheet_num, usecols="B,D,E")

    coords["Frame Number"] = coords["Frame Number"].ffill().astype(int)
    coords.set_index("Frame Number", inplace=True)

    assert coords.isnull().values.any() == 0

    flx_ext_pts = [117, 299, 630, 117]
    uqf = coords.index.unique()

    metadata = {"knee_id": sheet_names[sheet_num], "flx_ext_pt": flx_ext_pts[sheet_num], "f0": uqf[0], "fN": uqf[-1]}

    return coords, metadata


def load_tif(filename) -> np.ndarray:
    """
    Inputs:
        filename (str) - path to the grayscale .tif multi-image file to be loaded. 
        
    Outputs:
        video (np.ndarray) - 3-dim array (nframes, h, w) containing the video information.
    """

    if VERBOSE: print("load_tif() called!")

    video = tif_imread(filename) # Imports image stack as np.ndarray (3 dimensions)
    _, h, w = video.shape # Dimensions of video stack
    video = np.concatenate( (np.zeros((1,h,w), dtype=np.uint8),video), axis=0) # Prepend blank frame -> 1-based indexing
    
    return video

def save_nparray(array:np.ndarray, filepath:str) -> None:
    """Saves numpy array as .npy file"""
    if VERBOSE: print("save_nparray() called!")

    if not isinstance(array, np.ndarray):
        raise TypeError("save_nparray(): video must be a numpy array")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, array)
    return

def load_nparray(filepath:str) -> np.ndarray:
    """Loads numpy array from .npy file"""
    if VERBOSE: print("load_nparray() called!")

    if not os.path.exists(os.path.dirname(filepath)):
        raise FileNotFoundError("load_nparray(): specified file not found")
    array = np.load(filepath)

    return array

def load_avi(fn) -> np.ndarray:
    if VERBOSE: print("load_avi() called!")

    cap = cv2.VideoCapture(fn)
    video = []
    while cap.isOpened():

        ret, frame = cap.read() # Returns (boolean, bgr frame)
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        video.append(frame)

    cap.release()
    video = np.array(video)
    return video

def save_avi(filepath: str, video: np.ndarray, fps: int = 30) -> None:
    """Save a NumPy video array to disk as an .avi (MJPG) file.

    Parameters
    ----------
    filepath : str
        Target file path; relative paths are resolved against the CWD.
    video : np.ndarray
        Array of shape (n_frames, H, W)  or  (n_frames, H, W, 3).
        dtype uint8 preferred; float 0‑1 will be auto‑scaled.
    fps : int, optional
        Frames per second for the output file (default 30).
    """
    # Resolve path and create parent dirs
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Validate video shape
    if video.ndim not in (3, 4):
        raise ValueError("video must have shape (n, H, W) or (n, H, W, 3)")
    n_frames, h, w = video.shape[:3]
    is_color = video.ndim == 4

    # Ensure uint8
    if video.dtype != np.uint8:
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)

    # MJPG is broadly supported
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h), isColor=is_color)

    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {filepath}")

    for i in range(n_frames):
        frame = video[i]
        if is_color:
            # CV expects BGR; assume input is already BGR
            writer.write(frame)
        else:
            # For grayscale, VideoWriter still expects 3‑channel unless isColor=False
            writer.write(frame)

    writer.release()
    print(f"Saved {n_frames} frames to {filepath}")

def save_mp4(filepath: str, video: np.ndarray, fps: int = 30) -> None:
    """Save a NumPy video array to disk as an .mp4 (H.264) file.

    Parameters
    ----------
    filepath : str
        Target file path; relative paths are resolved against the CWD.
    video : np.ndarray
        Array of shape (n_frames, H, W)  or  (n_frames, H, W, 3).
        dtype uint8 preferred; float 0‑1 will be auto‑scaled.
    fps : int, optional
        Frames per second for the output file (default 30).
    """
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim not in (3, 4):
        raise ValueError("video must have shape (n, H, W) or (n, H, W, 3)")
    n_frames, h, w = video.shape[:3]
    is_color = video.ndim == 4

    # Normalize to uint8
    if video.dtype != np.uint8:
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)

    # Choose codec – mp4v is broadly supported for MP4 container
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h), isColor=is_color)

    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {filepath}")

    for i in range(n_frames):
        frame = video[i]
        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)

    writer.release()
    print(f"Saved {n_frames} frames to {filepath}")


def concatenate_avi(file1, file2, out_path="output_concat.avi"):
    """
    Horizontally concatenates two .avi video files, padding frames or lengths if necessary.

    Parameters
    ----------
    file1 : str
        Path to the first .avi video.
    file2 : str
        Path to the second .avi video.
    out_path : str
        Path to save the concatenated output video.
    """

    cap1 = cv2.VideoCapture(file1)
    cap2 = cv2.VideoCapture(file2)

    if not cap1.isOpened() or not cap2.isOpened():
        raise IOError("One of the video files could not be opened.")

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    length1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    max_length = max(length1, length2)

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    H = max(h1, h2)
    W = w1 + w2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    def read_frame(cap, default_size):
        ret, frame = cap.read()
        if not ret:
            return np.zeros(default_size, dtype=np.uint8)
        if frame.shape[0] < default_size[0] or frame.shape[1] < default_size[1]:
            padded = np.zeros(default_size, dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1], :] = frame
            return padded
        return frame

    for _ in range(max_length):
        frame1 = read_frame(cap1, (H, w1, 3))
        frame2 = read_frame(cap2, (H, w2, 3))

        # Pad height if needed
        if frame1.shape[0] < H:
            pad_h = H - frame1.shape[0]
            frame1 = np.vstack([frame1, np.zeros((pad_h, w1, 3), dtype=np.uint8)])
        if frame2.shape[0] < H:
            pad_h = H - frame2.shape[0]
            frame2 = np.vstack([frame2, np.zeros((pad_h, w2, 3), dtype=np.uint8)])

        concat = np.hstack((frame1, frame2))
        out.write(concat)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Saved to: {os.path.abspath(out_path)}")

import cv2
import os

def convert_avi_to_mp4(input_path: str, output_path: str = None, codec: str = 'mp4v') -> None:
    """
    Converts an .avi video to .mp4 using OpenCV.

    Parameters
    ----------
    input_path : str
        Path to the input .avi video.

    output_path : str, optional
        Path to save the .mp4 output. If None, replaces .avi with .mp4 in the input path.

    codec : str
        Four-character code of codec to use (default is 'mp4v').
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved MP4 to {output_path}")



def convert_avi_to_hdf5_grayscale(
    avi_path: str,
    output_dir: str,
    chunk_size: int = 200,
    overwrite: bool = True,
    verbose: bool = True
) -> str:
    """
    Converts a color .avi video to grayscale and stores it in HDF5 format.

    Parameters
    ----------
    avi_path : str
        Path to the input .avi video file.
    output_dir : str
        Target directory to save the .h5 file.
    chunk_size : int, optional
        Number of frames per chunk in the HDF5 dataset (default is 200).
    overwrite : bool, optional
        Whether to overwrite an existing .h5 file (default True).
    verbose : bool, optional
        Whether to print progress messages (default True).

    Returns
    -------
    str
        Path to the created HDF5 file.
    """
    avi_path = Path(avi_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / (avi_path.stem + "_grayscale.h5")

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} already exists and overwrite=False.")

    if verbose:
        print(f"Opening video: {avi_path}")

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {avi_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if verbose:
        print(f"Video has {n_frames} frames of size ({h}, {w})")

    # Prepare HDF5 file
    with h5py.File(output_file, "w") as h5f:
        print("Creating .hdf5 file...")
        dset = h5f.create_dataset(
            "video",
            shape=(n_frames, h, w),
            dtype=np.uint8,
            chunks=(chunk_size, h, w),
            compression="gzip"
        )

        print("Reading video...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print(frame_idx)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dset[frame_idx] = gray
            frame_idx += 1

            if verbose and frame_idx % 200 == 0:
                print(f"Processed {frame_idx}/{n_frames} frames...")

        cap.release()

    if verbose:
        print(f"Saved grayscale video to: {output_file}")

    return str(output_file)

def load_hdf5_video_chunk(
    h5_path: str,
    frame_slice: Optional[Tuple[int, int]] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Loads a chunk or the entire grayscale video stored in HDF5 format.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing the video dataset under "video".
    frame_slice : tuple(int, int), optional
        Tuple (start_frame, end_frame) specifying frames to load (inclusive start, exclusive end).
        If None, loads entire video.
    verbose : bool, optional
        Whether to print loading info (default False).

    Returns
    -------
    np.ndarray
        Loaded grayscale video chunk with shape (n_frames, height, width).
    """
    h5_path = Path(h5_path).expanduser().resolve()

    if verbose:
        print(f"Opening HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        if "video" not in h5f:
            raise KeyError(f"'video' dataset not found in {h5_path}")

        dset = h5f["video"]
        total_frames = dset.shape[0]

        if frame_slice is None:
            start, end = 0, total_frames
        else:
            start, end = frame_slice
            if start < 0 or end > total_frames or start >= end:
                raise ValueError(f"Invalid frame_slice {frame_slice} for dataset with {total_frames} frames")

        if verbose:
            print(f"Loading frames {start} to {end-1} (total {end-start})")

        video_chunk = dset[start:end]

    return np.array(video_chunk)
