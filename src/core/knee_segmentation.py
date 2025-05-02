import os
import sys
import numpy as np
import pandas as pd
import cv2
from tifffile import imread as tif_imread
from src.utils import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

VERBOSE = True
DEBUG = True

def load_aging_knee_coords(filename:str, knee_name:str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Inputs:
        filename (str) - path to the .xlsx coordinates file to be loaded
        sheet_sel (int) - index of the Excel sheet to be used 

    Outputs:
        coords (pd.DataFrame) - contains the pairs of coordinates provided by Huizhu @ Fudan University
        metadata (Dict) - contains information mostly relevant for plotting. See keys for info.
    """
    
    if VERBOSE: print("load_aging_knee_coords() called!")

    # Import knee coordinates
    coords_file = pd.read_excel(filename, engine='openpyxl', sheet_name=None) # More updated Excel import
    # coords_file = pd.read_excel("../data/xy coordinates for knee-aging three cycles 250303.xlsx", engine='openpyxl', sheet_name=None) # More updated Excel import
    # coords_file = pd.read_excel("../data/adjusted xy coordinates for knee-aging 250403.xlsx", engine='openpyxl', sheet_name=None) # More updated Excel import

    # Select data set
    # knee_opts = ['aging-1', 'aging-2', 'aging-3']
    # knee_name = knee_opts[sheet_sel]
    coords_sheet = coords_file[knee_name] # Set index = {0,1,2} to choose different data set

    # Clean data
    coords_sheet.drop(columns=['Unnamed: 0', 'Unnamed: 5'], axis=1, inplace=True) # No information

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

    metadata = {"knee_name": knee_name, "flx_ext_pt": flx_ext_pt, "f0": uqf[0], "fN": uqf[-1]}
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

def pre_process_video(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        video (np.ndarray) - video to be pre-processed. Dimensions (nframes, height, width)

    Outputs:
        video (np.ndarray) - processed video. Dimensions (nframes, height, width)
        translation_mxs (np.ndarray) - translation matrices used to centre each frame
    """
    if VERBOSE: print("pre_process_video() called!")

    video_ctrd = []
    translation_mxs = []
    for idx, frame in enumerate(video):

        # Process frame
        frame, tr_mx = utils.centroid_stabilization(frame)

        # Store data
        video_ctrd.append(frame)
        translation_mxs.append(tr_mx)

    video_ctrd = np.array(video_ctrd)
    translation_mxs = np.array(translation_mxs)
    return video_ctrd, translation_mxs

def translate_coords(translation_mxs: np.ndarray, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs: 
        translation_mxs (np.ndarray) - 2x3 translation matrices
        coords (pd.DataFrame) - df of coordinates to be transformed. Expected format: coords.index := frame number; four points per frame
    Outputs: 
        coords_ctrd (pd.DataFrame) - df of transformed coordinates
    """
    if VERBOSE: print("translate_coords() called!")

    coords_ctrd = pd.DataFrame(np.nan, index=coords.index, columns=coords.columns) # empty dataframe
    uqf = coords.index.unique()
    for cf in uqf:
        
        # Apply translations to coords
        tr_mx = translation_mxs[cf]
        xp = np.row_stack([coords.loc[cf].to_numpy().T, np.ones(4)])
        coord_ctrd = tr_mx @ xp

        # Store result
        coords_ctrd.loc[cf] = coord_ctrd.T

    return coords_ctrd

def get_three_segments(video: np.ndarray, coords: np.ndarray, thresh_scale:int=0.8) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Inputs: 
        video (np.ndarray) - video to be manually segmented
        coords (np.ndarray) - coordinates to use for three part segmentation
        thresh_scale(int) - to rescale the Otsu threshold mask 
    Outputs:
        regions (Dict[str, np.ndarray]) - segmented regions obtained 
        masks (Dict[str, np.ndarray]) - segmented masks obtained
    """
    if VERBOSE: print("get_three_segments() called!")

    video = video.copy()

    otsu_masks = []
    for cf, frame in enumerate(video):
        
        # Get otsu mask
        thresh_val, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(thresh_val*thresh_scale)
        _, otsu_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Store otsu mask
        otsu_masks.append(otsu_mask)
        
    otsu_masks = np.array(otsu_masks)


    l_masks = []
    m_masks = []
    r_masks = []
    l_region = []
    m_region = []
    r_region = []
    otsu_region = []
    for cf in coords.index.unique():

        frame = video[cf]        
        cf_coords = coords.loc[cf].to_numpy().astype(int)

        # Get rough bisection mask
        mp0 = (cf_coords[0]+cf_coords[2])//2 # top 
        mp1 = (cf_coords[1]+cf_coords[3])//2 # top 
        lr_mask = utils.pixels_left_of_line(frame, mp1, mp0)

        # Get rough middle mask
        _m_mask_l = utils.pixels_left_of_line(frame, cf_coords[0], cf_coords[1])
        _m_mask_r = utils.pixels_left_of_line(frame, cf_coords[3], cf_coords[2])
        m_mask = _m_mask_l & _m_mask_r

        # Get rough left and right masks
        l_mask = lr_mask & ~m_mask
        r_mask = ~lr_mask & ~m_mask

        # Get final masks
        otsu_mask = otsu_masks[cf]
        l_mask = l_mask & otsu_mask
        m_mask = m_mask & otsu_mask
        r_mask = r_mask & otsu_mask

        # Get l/m/r/Otsu regions
        l_reg = l_mask & frame
        m_reg = m_mask & frame
        r_reg = r_mask & frame
        otsu_reg = otsu_mask & frame

        # Store vals
        l_masks.append(l_mask)
        m_masks.append(m_mask)
        r_masks.append(r_mask)

        l_region.append(l_reg)
        m_region.append(m_reg)
        r_region.append(r_reg)
        otsu_region.append(otsu_reg)

    # Cast to numpy arrays
    l_masks = np.array(l_masks)
    m_masks = np.array(m_masks)
    r_masks = np.array(r_masks)
    l_region = np.array(l_region)
    m_region = np.array(m_region)
    r_region = np.array(r_region)
    otsu_region = np.array(otsu_region)

    # Store in dict
    masks = {"l": l_masks, "m": m_masks, "r": r_masks, "otsu": otsu_masks}
    regions = {"l": l_region, "m": m_region, "r": r_region, "otsu": otsu_region}
    
    return regions, masks

def _measure_region_intensity(region: np.ndarray) -> np.ndarray:
    """
        Helper function. Returns sum of pixel intensities for every frame
    """
    # if VERBOSE: print("_measure_region_intensity() called!")

    intensities = []
    for cf, frame in enumerate(region):
        intsty = np.sum(frame)
        intensities.append(intsty)

    intensities = np.array(intensities)

    return intensities

# TODO: refine return type hint. TypedDict? 
def measure_region_intensities(regions: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], keys: List[str], normalized=False) -> Dict: 
    """
    Inputs: 
        regions (Dict[str, np.ndarray]) - segmented regions for which brightness is to be measured
        masks (Dict[str, np.ndarray]) - corresponding segmented masks used for normalization
        keys (List[str]) - specifies which regions are to be measured
        normalized (bool) - flag to normalize data
    Outputs:
        region_intensities (Dict[str, np.ndarray, bool]) - Dict of region intensity readings (np.ndarray); and also a normalized (bool) flag
    """
    if VERBOSE: print("measure_region_intensities() called!")

    if normalized:
        if VERBOSE: print(" > normalized!")
        mask_intensities = {}
        for k in keys:
            mask_intensities[k] = _measure_region_intensity(masks[k])

    region_intensities = {}
    for k in keys:
        region_intensities[k] = _measure_region_intensity(regions[k])
        
        if normalized:
            region_intensities[k] = region_intensities[k] / mask_intensities[k]
    
    region_intensities["normalized"] = normalized # Store some metadata

    return region_intensities

def plot_three_intensities(intensities: Dict, metadata: Dict, show_figs=True, save_figs=False) -> None:
    """
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensity values to be plotted 
        metadata (Dict[str, str, int]) - plotting metadata from load_aging_knee_coords()
        show_figs (bool) - show or hide figs. Default is True
        save_figs (bool) - save or don't safe figs to standard "figures/" directory. Default is False
    """
    if VERBOSE: print("plot_three_intensities() called!")

    normalized = intensities["normalized"] # Get intensity metadata
    keys = ["l", "m", "r"] # Hardcoded 
    plt.style.use('default')

    # Prepare formatting strings
    if normalized: ttl_sfx = "(Normalized " + metadata["knee_name"] + ")"
    else: ttl_sfx = "(Raw " + metadata["knee_name"] + ")"    
    ttl_pfx = {"l": "Left", "m": "Middle", "r": "Right", "otsu": "Whole"}
    clrs = {"l": "r", "m": "g", "r": "b", "otsu": NotImplemented}
    if normalized: sv_fl_pfx = "normalized"
    else: sv_fl_pfx = "raw"

    # Plot three (or more) figs separately
    fig, axes = plt.subplots(1, len(keys), figsize=(20,7))
    i = 0
    for k in keys:

        # Plot intensities
        fns = np.arange(metadata["f0"], metadata["f0"] + len(intensities[k]))
        axes[i].plot(fns, intensities[k], color=clrs[k])

        # Formatting
        axes[i].set_title(ttl_pfx[k] + " knee pixel intensities " + ttl_sfx)
        axes[i].axvline(metadata["flx_ext_pt"], color="k", linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
        axes[i].legend()

        i+=1

    if save_figs:
        fn = f"../figures/intensity_plots/{sv_fl_pfx}_separate_{metadata['knee_name']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    # plt.show()


    # Plot three (or more) figs combined
    plt.figure(figsize=(15,8))
    for k in keys:
        plt.plot(fns, intensities[k], color=clrs[k], label=ttl_pfx[k] + " knee")
    plt.axvline(metadata["flx_ext_pt"], color='k', linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
    plt.title("Knee pixel intensities " + ttl_sfx)
    plt.legend()

    if save_figs:
        fn = f"../figures/intensity_plots/{sv_fl_pfx}_combined_{metadata['knee_name']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    if show_figs:
        plt.tight_layout()
        plt.show()

    return


def get_intensity_diffs(intensities: Dict) -> Dict[str, np.ndarray]:
    """
    Returns the difference in intensity values between every frame (backward difference). 
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensities for which derivatives are to be obtained
    Outputs:
        derivs (Dict[np.ndarray]) - derivatives obtained
    """
    if VERBOSE: print("get_intensity_diffs() called!")

    keys = ['l','m','r']
    derivs = {}
    for k in keys:
        deriv = pd.Series(intensities[k].astype(np.int64)).diff() # Cast from uint32 
        derivs[k] = deriv
    return derivs


def display_regions(regions:Dict[str, np.ndarray], keys:List[str]) -> None:
    if VERBOSE: print("display_regions() called!")

    n_frames = regions[keys[0]].shape[0] # Assume each np.ndarray in regions[] has the same dimensions
    for cf in np.arange(n_frames):
        f_stack = np.hstack(tuple([utils.crop_square_frame(regions[k][cf], n=350) for k in keys]))
        cv2.imshow(f"display_regions()", f_stack)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

def smooth_coords(coords:pd.DataFrame, window_size:int) -> pd.DataFrame:
    """Implements a moving average filter over the coordinate data."""
    if VERBOSE: print("smooth_coords() called!")

    coords_smtd = coords.copy()
    coords_smtd['X'] = coords_smtd["X"].rolling(window_size, min_periods=1).mean()
    coords_smtd['Y'] = coords_smtd["Y"].rolling(window_size, min_periods=1).mean()
    
    return coords_smtd

def plot_coords(video:np.ndarray, coords:pd.DataFrame) -> None:
    if VERBOSE: print("plot_coords() called!")

    video = video.copy()
    uqf = coords.index.unique()
    for cf in uqf:

        frame = video[cf]
        pts = coords.loc[cf].to_numpy().astype(int)

        for x,y in pts:
            cv2.circle(frame, (x,y), 3, (255,255,255))

        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255,255,255), 1)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (255,255,255), 1)

        cv2.imshow("plot_coords()", frame)

        if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()    
    return

def smooth_coords(coords:pd.DataFrame, window_size:int) -> pd.DataFrame:
    """Implements a moving average filter over the coordinate data."""
    if VERBOSE: print("smooth_coords() called!")

    assert coords.shape[0]%4 == 0
    nrows=coords.shape[0]//4

    p1 = coords.iloc[0::4, :].copy()
    p2 = coords.iloc[1::4, :].copy()
    p3 = coords.iloc[2::4, :].copy()
    p4 = coords.iloc[3::4, :].copy()

    ps = [p1, p2, p3, p4]
    for p in ps:
        p["X"] = p["X"].rolling(window_size, min_periods=1, center=True).mean()
        p["Y"] = p["Y"].rolling(window_size, min_periods=1, center=True).mean()
    
    coords_smtd = []
    for r in range(nrows):
        for p in ps:
            coords_smtd.append(p.iloc[r])
    coords_smtd = pd.DataFrame(coords_smtd)

    return coords_smtd

def get_intensity_derivs(intensities:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    """
    Implements a second-order finite difference approximation for the first derivative of the intensity data
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensities for which derivatives are to be obtained
    Outputs:
        derivs (Dict[np.ndarray]) - derivatives obtained
    """
    if VERBOSE: print("get_intensity_derivs() called!")

    keys = ['l','m','r']
    derivs = {}
    for k in keys:
        derivs[k] = np.gradient(intensities[k], edge_order=2)
    return derivs

def view_frames(video:np.ndarray) -> None:
    """Shows all frames. Press any button to advance, or 'q' to exit"""
    if VERBOSE: print("view_frames() called!")

    for cf, frame in enumerate(video):
        cv2.imshow("view_frames()", frame)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

    return
