import os
import sys
module_path = os.path.abspath(os.path.join('..', 'utils')) # Build an absolute path from this notebook's parent directory
if module_path not in sys.path: # Add to sys.path if not already present
    sys.path.append(module_path)
import numpy as np
import pandas as pd
import cv2
from tifffile import imread as tif_imread
import utils
from typing import Tuple, Dict, List

DATA_IDX = 2
MODIFY_DATA = True 
GENERATE_FIGURES = False
LOOP = False
VERBOSE = True

def load_knee_coords(filename:str, sheet_sel:int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Inputs:
        filename (str) - path to the .xlsx coordinates file to be loaded
        sheet_sel (int) - index of the Excel sheet to be used 

    Outputs:
        coords (pd.DataFrame) - contains the pairs of coordinates provided by Huizhu @ Fudan University
        knee_name (str) - the name of the selected Excel sheet    
        flx_ext_pt (int) - the midpoint of the flexion/extension cycle
    """
    
    if VERBOSE: print("load_knee_coords() called!")

    # Import knee coordinates
    coords_file = pd.read_excel(filename, engine='openpyxl', sheet_name=None) # More updated Excel import
    # coords_file = pd.read_excel("../data/xy coordinates for knee-aging three cycles 250303.xlsx", engine='openpyxl', sheet_name=None) # More updated Excel import
    # coords_file = pd.read_excel("../data/adjusted xy coordinates for knee-aging 250403.xlsx", engine='openpyxl', sheet_name=None) # More updated Excel import

    # Select data set
    knee_opts = ['aging-1', 'aging-2', 'aging-3']
    knee_name = knee_opts[sheet_sel]
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

    metadata = {"knee_name": knee_name, "flx/ext_pt": flx_ext_pt}
    return coords, metadata

def load_tif(filename):
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

def process_video(video: np.ndarray, coords: np.ndarray, knee_name: str):
    
    if VERBOSE: print("process_video() called!")

    # Get unique frames
    unique_frames = coords.index.unique()

    # Get basic frame info
    first_frame = unique_frames[0]
    last_frame = unique_frames[-1]
    curr_frame = first_frame

    # Initialize data storage
    knee_intensities = np.zeros((3, unique_frames.shape[0]))
    knee_intensities_normalized = np.zeros((3, unique_frames.shape[0]))
    knee_total_areas = np.zeros((3, unique_frames.shape[0])) # Save total area of each mask 
    frames_out = []
    while curr_frame <= last_frame:

        # Get frame
        frame = video[curr_frame, :, :].copy() # true copy

        # Pre-processing
        frame, translation_mx = utils.centroid_stabilization(frame) # Center the frame based on the 


        'Coordinates plotting'

        # Get coordinates to be plotted
        cf_coords = coords.loc[curr_frame]

        # Transform coordinates according to centroid stabilization
        cf_coords = np.column_stack([cf_coords.to_numpy(), np.ones(cf_coords.shape[0])])
        cf_coords = (translation_mx @ cf_coords.T).T[:, 0:2]

        # Store coordinates as integer tuples
        point_coords = [tuple(cf_coords[pt_n].astype(int)) for pt_n in range(4)]

        # Plot points
        # for x, y in point_coords:
        #     cv2.circle(frame, (x, y), 3, [255, 255, 255], -1)

        # Draw lines between points 0-1 and 2-3
        # cv2.line(frame, point_coords[0], point_coords[1], [255, 255, 255], 1)
        # cv2.line(frame, point_coords[2], point_coords[3], [255, 255, 255], 1)
        # cv2.circle(frame, point_coords[3], 3, [255, 255, 255], -1)

        # Draw line between left and right region
        midpoints = [
        ((point_coords[0][0] + point_coords[2][0]) // 2, (point_coords[0][1] + point_coords[2][1]) // 2),
        ((point_coords[1][0] + point_coords[3][0]) // 2, (point_coords[1][1] + point_coords[3][1]) // 2)
        ]
        # cv2.line(frame, midpoints[0], midpoints[1], [255,0,0], 1)


        'Three-parts segmentation'

        # Get left/right mask based on the middle line
        left_right_mask = utils.pixels_left_of_line(frame, midpoints[1], midpoints[0])
        left_right_mask = utils.crop_square_frame(left_right_mask, 350) # For ease of viewing

        # Get middle mask between the left and right lines
        left_mask = utils.pixels_left_of_line(frame, point_coords[1], point_coords[0])
        right_mask = utils.pixels_left_of_line(frame, point_coords[2], point_coords[3])
        left_mask = utils.crop_square_frame(left_mask, 350) # For ease of viewing
        right_mask = utils.crop_square_frame(right_mask, 350) # For ease of viewing
        middle_mask = ~left_mask & ~right_mask

        # Get Otsu's mask
        thresh_val, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(thresh_val*0.8)
        _, otsu_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        otsu_mask = utils.crop_square_frame(otsu_mask, 350) # For ease of viewing

        # Get knee masks
        middle_knee_mask = otsu_mask & middle_mask
        left_knee_mask = otsu_mask & left_right_mask & ~middle_mask #& left_mask
        right_knee_mask = otsu_mask & ~left_right_mask & ~middle_mask #& right_mask

        # Crop frame to square region
        frame = utils.crop_square_frame(frame, 350) # For ease of viewing

        # Resize frame for easier viewing
        pixel_scale = 10.1119 # pixels / mm
        # frame, scaling_factor = utils.rescale_frame(frame, pixel_scale, 0)


        'Get sum of pixel intensities in each part'
        
        # Get left/middle/right knee
        left_knee = frame & left_knee_mask
        middle_knee = frame & middle_knee_mask
        right_knee = frame & right_knee_mask

        # Get total number of non-zero pixels
        left_knee_nonzero = np.sum(left_knee_mask)
        middle_knee_nonzero = np.sum(middle_knee_mask)
        right_knee_nonzero = np.sum(right_knee_mask)

        # Get normalized left/middle/right knee pixel intensities
        curr_idx = curr_frame - first_frame
        if MODIFY_DATA:
            knee_intensities[0, curr_idx] = np.sum(left_knee) # left intensity
            knee_intensities[1, curr_idx] = np.sum(middle_knee) # middle intensity
            knee_intensities[2, curr_idx] = np.sum(right_knee) # right intensity

            knee_intensities_normalized[0, curr_idx] = np.sum(left_knee) / left_knee_nonzero # left intensity
            knee_intensities_normalized[1, curr_idx] = np.sum(middle_knee) / middle_knee_nonzero # middle intensity
            knee_intensities_normalized[2, curr_idx] = np.sum(right_knee) / right_knee_nonzero # right intensity    

            knee_total_areas[0, curr_idx] = np.sum(left_knee_nonzero) # left mask sum
            knee_total_areas[1, curr_idx] = np.sum(middle_knee_nonzero) # middle mask sum
            knee_total_areas[2, curr_idx] = np.sum(right_knee_nonzero) # right mask sum


        'Display frames and information'

        # --- For demonstration: overlay the masks on the frame ---
        # frame = np.maximum.reduce([frame, left_knee_mask, middle_knee_mask, right_knee_mask])

        # Show knee_mask, to verify boundaries and such
        knee_mask = np.maximum.reduce([left_knee, middle_knee, right_knee])

        # Write frame number in bottom left corner of knee_mask
        h,w = knee_mask.shape
        pos = (10, h - 10)
        cv2.putText(knee_mask, str(curr_frame), pos, 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, 
                    color = (255,255,255), thickness = 2, lineType = cv2.LINE_AA)
        # cv2.imshow(f"knee_mask, frames {first_frame} to {last_frame}", knee_mask) # Display knee_mask

        # Write coordinates in top-left corner 
        pos1 = (int(3*w//4), 15)
        lines = [tuple(coords.loc[curr_frame].iloc[i].astype(int)) for i in range(0,4)]
        cv2.putText(knee_mask, f"l line: {lines[0]}", pos1, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, 
                color = (255,255,255), thickness = 1, lineType = cv2.LINE_AA)
        pos2 = (int(3*w//4), 30)
        cv2.putText(knee_mask, f"       {lines[1]}", pos2, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, 
                color = (255,255,255), thickness = 1, lineType = cv2.LINE_AA)
        pos3 = (int(3*w//4), 45)
        cv2.putText(knee_mask, f"r line: {lines[2]}", pos3, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, 
                color = (255,255,255), thickness = 1, lineType = cv2.LINE_AA)
        pos4 = (int(3*w//4), 60)
        cv2.putText(knee_mask, f"       {lines[3]}", pos4, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, 
                color = (255,255,255), thickness = 1, lineType = cv2.LINE_AA)


        'Miscellaneous technical things'

        # Nice display
        frame_out = cv2.hconcat([knee_mask, left_knee_mask, middle_knee_mask, right_knee_mask])
        # cv2.imshow(f"{knee_name} knee (frames {first_frame}-{last_frame})", frame_out)

        frames_out.append(frame_out)


        if GENERATE_FIGURES:
            fn = f"../figures/labeled {knee_name} frames/{knee_name}_{curr_frame:04d}.png"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(fn, frame_out) 


        # Increment frame index
        curr_frame += 1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        # Optionally LOOP video
        if LOOP and curr_frame > last_frame: 
            curr_frame = first_frame

    cv2.destroyAllWindows()
    frames_out = np.array(frames_out)
    return frames_out


def pre_process_video(video):
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
    if VERBOSE: print("translate_coords() called!")

    coords_ctrd = pd.DataFrame(np.nan, index=coords.index, columns=coords.columns) # empty dataframe
    uniq_f = coords.index.unique()
    for cf in uniq_f:
        
        # Apply translations to coords
        tr_mx = translation_mxs[cf]
        xp = np.row_stack([coords.loc[cf].to_numpy().T, np.ones(4)])
        coord_ctrd = tr_mx @ xp

        # Store result
        coords_ctrd.loc[cf] = coord_ctrd.T

    return coords_ctrd

def get_three_masks(video: np.ndarray, coords: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if VERBOSE: print("get_three_masks() called!")

    video = video.copy()

    otsu_masks = [] # TODO: abstract this to another function?
    for cf, frame in enumerate(video):
        
        # Get otsu mask
        thresh_val, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(thresh_val*0.8) # TODO: parameterize hardcoded 20% decrease 
        _, otsu_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Store otsu mask
        otsu_masks.append(otsu_mask)
        
    otsu_masks = np.array(otsu_masks)


    l_masks = []
    m_masks = []
    r_masks = []
    for cf in coords.index.unique():

        frame = video[cf]        
        cf_coords = coords.loc[cf].to_numpy().astype(int)

        # Get bisection mask
        mp0 = (cf_coords[0]+cf_coords[2])//2 # top 
        mp1 = (cf_coords[1]+cf_coords[3])//2 # top 
        lr_mask = utils.pixels_left_of_line(frame, mp1, mp0)

        # Get middle mask
        _m_mask_l = utils.pixels_left_of_line(frame, cf_coords[0], cf_coords[1])
        _m_mask_r = utils.pixels_left_of_line(frame, cf_coords[3], cf_coords[2])
        m_mask = _m_mask_l & _m_mask_r

        # Get left and right masks
        l_mask = lr_mask & ~m_mask
        r_mask = ~lr_mask & ~m_mask

        # Get final masks
        l_mask = l_mask & otsu_masks[cf]
        m_mask = m_mask & otsu_masks[cf]
        r_mask = r_mask & otsu_masks[cf]

        # Store vals
        l_masks.append(l_mask)
        m_masks.append(m_mask)
        r_masks.append(r_mask)

    l_masks = np.array(l_masks)
    m_masks = np.array(m_masks)
    r_masks = np.array(r_masks)

    masks = {"l": l_masks, "m": m_masks, "r": r_masks, "otsu": otsu_masks}
    
    return None, masks

def measure_mask_intensities(masks: Dict[str, np.ndarray], keys: List[str], normalized=False) -> Dict[str, np.ndarray]:

    return None

# Intended code execution path:
# > Load video
# > Load coords 
# > Centre video 
# > Centre coords
# > Get masks
# x Process data
# x Plot data 

def main():
    if VERBOSE: print("main() called!")

    # Load data and metadata
    video = load_tif("../data/1 aging_00000221.tif")
    sheet_sel = 2
    coords, metadata = load_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", sheet_sel) # TODO: load all data at once and pass coords as a dict?

    # Pre-process data (centroid stabilization)
    video_ctrd, translation_mxs = pre_process_video(video)
    coords_ctrd = translate_coords(translation_mxs, coords)

    # Get masks
    regions, masks = get_three_masks(video_ctrd, coords_ctrd) # Returns a dict of masks

    # Get intensity data
    # keys = ["l", "m", "r"]
    # raw_intensities = measure_mask_intensities(masks, keys) # TODO: Returns a dict of intensity measurements 
    # normalized_intensities = measure_mask_intensities(masks, keys, normalized=True)


if __name__ == "__main__":
    main()

