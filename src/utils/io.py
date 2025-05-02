import pandas as pd
import numpy as np
import os
from tifffile import imread as tif_imread
from src.config import VERBOSE
from typing import Tuple, Dict, Union


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