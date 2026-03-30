"""
This script will go through all the video files for which an Excel file exists, and verify that the information contained in the Excel file matches the video file.
"""

from typing import List, Tuple
from utils import io
import core.data_processing as dp
import config.knee_metadata as knee_meta
import pandas as pd
import os

def discover_excel_files() -> List[str]:
    """
    Discovers all Excel files in the data/intensities_total directory.
    Returns a list of filenames
    """

    if not os.path.isdir("data/intensities_total"):
        return []

    filenames = []
    for name in os.listdir("data/intensities_total"):
        if name.endswith(".xlsx"):
            filenames.append(name)

    return filenames


def discover_video_files() -> List[str]:
    """
    Discovers all video files in the data/segmented directory.
    Returns a list of filenames
    """
    if not os.path.isdir("data/segmented"):
        return []

    video_files = []
    for name in os.listdir("data/segmented"):

        if not name.endswith(".npy"):
            continue

        try: cond, id, dtype, nsegs = name.split("_")
        except: continue

        if dtype == "video":
            video_files.append(name)

    return video_files


def resolve_video_files(excel_files: List[str]) -> Tuple[List[str], List[str]]:
    """
    Discovers the video files corresponding to the Excel files.
    Returns two lists: (video_files, mask_files),
    """

    video_files = discover_video_files()

    for file in excel_files:
        id = file[0:4]
        nsegs = file[5:7]

        for video in video_files: 
            if not id in video or not nsegs in video: 
                video_files.remove(video)
                break
        

    return video_files


def verify_intensity_data(excel_files: List[str], video_files: List[str]) -> None:
    """
    Verifies the intensity data in the Excel files against the video files.
    """
    # Compare the intensity data using dp.compute_sums_nonzeros
    # If there is a mismatch, print a warning

    return

def verify_metadata() -> None:
    """
    Verifies the metadata in the Excel files against the metadata in the knee_metadata module.
    """
    # Get all the metadata from the knee_metadata module
    # Get all the metadata from the Excel files
    # Compare the metadata
    # If there is a mismatch, print a warning


def main():

    # Discover all Excel filenames
    excel_files = discover_excel_files()

    # Resolve corresponding video filenames
    video_files = resolve_video_files(excel_files)

    # Verify intensity data and metadata
    for excel, video, mask in zip(excel_files, video_files, mask_files):
        verify_intensity_data(excel, video, mask)
        verify_metadata(excel)

    return


if __name__ == "__main__":
    main()


