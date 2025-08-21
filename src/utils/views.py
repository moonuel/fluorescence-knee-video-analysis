import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from config import VERBOSE
from typing import Dict, List, Tuple
from utils import utils
from pathlib import Path


def plot_coords(video:np.ndarray, coords:pd.DataFrame, title:str=None, show_video:bool=True) -> np.ndarray:
    """Plots the set of coordinates for the three part segmentation"""
    if VERBOSE: print("plot_coords() called!")

    video = video.copy()
    nfs, h, w = video.shape
    uqf = coords.index.unique()
    cf = uqf[0]

    btm_l_pos = (10, h - 10)

    if title is None: title = "plot_coords()"

    # while True:
    for cf in np.arange(uqf[0], uqf[-1] + 1):

        frame = video[cf]
        pts = coords.loc[cf].to_numpy().astype(int)

        # Draw points and lines
        for x,y in pts:
            cv2.circle(frame, (x,y), 3, (255,255,255))

        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255,255,255), 1)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (255,255,255), 1)

        # Draw line
        cv2.putText(frame, str(cf), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 0.7, color = (255, 255, 255), thickness = 1, lineType=cv2.LINE_AA)

        # Commit changes
        video[cf] = frame

        # cv2.imshow(title, frame)

        # # Controls
        # k = cv2.waitKey(0)
        # if k == ord('q'): break
        # if k == ord('a'): cf-=1
        # if k == ord('d'): cf+=1
        # cf = (cf - uqf[0]) % (uqf[-1] - uqf[0] + 1) + uqf[0] # Wrap cfs

    # cv2.destroyAllWindows()    

    if show_video: show_frames(video, title=title)

    return video

def plot_radial_segment_intensities(intensities:np.ndarray, f0:int=None, fN:int=None, 
                                    show_figs:bool=True, save_figs:bool=False, vert_layout:bool=False, figsize:tuple=(20,7)) -> Tuple:
    """
    Alternate function intended for use with the radial segmentation scheme.
    Plots all provided intensity figures.

    Pass f0 = fN = None for default settings (whole video).

    Returns (fig, axes)
    
    Args:
        intensities (np.ndarray): shape (n_slices, n_frames, h, w), intensity data
        f0 (int): starting frame index (inclusive), defaults to 0
        fN (int): ending frame index (inclusive), defaults to last frame
        show_figs (bool): whether to show figures
        save_figs (bool): whether to save figures to ../figures/
        vert_layout (bool): vertical or horizontal subplots
        figsize (tuple): figure size
    """
    if VERBOSE: print("plot_radial_segment_intensities() called!")

    nslcs, nfrms = intensities.shape

    if f0 is None: f0 = 0
    if fN is None: fN = nfrms

    intensities = intensities[:, f0 : fN+1].copy()

    frmns = np.arange(f0, fN)

    # Plot all slices on same figure
    plt.figure(figsize=figsize)
    for slc in range(nslcs):
        plt.plot(frmns, intensities[slc], label=f"Slice {slc}")
    plt.title("Radial Segment Intensities")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()

    if save_figs:
        save_path = "../figures/radial_intensity_combined.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_figs:
        plt.show()
    else:
        plt.close()

    # Plot each slice on its own subplot
    if vert_layout:
        fig, axes = plt.subplots(nslcs, 1, figsize=(figsize[0], figsize[1] * nslcs), squeeze=False)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, nslcs, figsize=(figsize[0] * nslcs, figsize[1]), squeeze=False)
        axes = axes.flatten()

    for slc in range(nslcs):
        axes[slc].plot(frmns, intensities[slc], label=f"Slice {slc}")
        axes[slc].set_title(f"Slice {slc} Intensity")
        axes[slc].set_xlabel("Frame")
        axes[slc].set_ylabel("Intensity")
        axes[slc].legend()

    plt.tight_layout()

    if save_figs:
        save_path = "../figures/radial_intensity_subplots.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_figs:
        plt.show()
    else:
        plt.close()

    return fig, axes

def plot_radial_segment_intensities_2(intensities: np.ndarray, f0: int = None, fN: int = None, 
                                      show_figs: bool = True, save_dir: str = None, 
                                      vert_layout: bool = False, figsize: Tuple[int, int] = (20, 7)) -> Tuple:
    """
    Different version so I don't break existing functions. Accepts 0-based frame indexing.
    Alternate function intended for use with the radial segmentation scheme.
    Plots all provided intensity figures. 

    Pass f0 = fN = None for default settings (whole video).

    Returns (fig, axes)
    
    Args:
        intensities (np.ndarray): shape (n_slices, n_frames), intensity data
        f0 (int): starting frame index (inclusive), defaults to 0
        fN (int): ending frame index (inclusive), defaults to last frame
        show_figs (bool): whether to show figures
        save_dir (str): if provided, saves the figure to this directory
        vert_layout (bool): vertical or horizontal subplots
        figsize (tuple): figure size
    """
    if VERBOSE: print("plot_radial_segment_intensities_2() called!")

    nslcs, nfrms = intensities.shape

    if f0 is None:
        f0 = 0
    if fN is None:
        fN = nfrms

    intensities = intensities[:, f0:fN].copy()
    frmns = np.arange(f0, fN)

    # === Plot all slices on same figure ===
    plt.figure(figsize=figsize)
    for slc in range(nslcs):
        plt.plot(frmns, intensities[slc], label=f"Slice {slc}")
    plt.title("Radial Segment Intensities")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir).expanduser().resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        filepath_combined = save_path / "radial_intensity_combined.png"
        plt.savefig(filepath_combined, dpi=300, bbox_inches="tight")
        print(f"Saved combined intensity plot to {filepath_combined}")

    if show_figs:
        plt.show()
    else:
        plt.close()

    # === Plot each slice on its own subplot ===
    if vert_layout:
        fig, axes = plt.subplots(nslcs, 1, figsize=(figsize[0], figsize[1] * nslcs), squeeze=False)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, nslcs, figsize=(figsize[0] * nslcs, figsize[1]), squeeze=False)
        axes = axes.flatten()

    for slc in range(nslcs):
        axes[slc].plot(frmns, intensities[slc], label=f"Slice {slc}")
        axes[slc].set_title(f"Slice {slc} Intensity")
        axes[slc].set_xlabel("Frame")
        axes[slc].set_ylabel("Intensity")
        axes[slc].legend()

    plt.tight_layout()

    if save_dir is not None:
        filepath_subplots = save_path / "radial_intensity_subplots.png"
        plt.savefig(filepath_subplots, dpi=300, bbox_inches="tight")
        print(f"Saved subplot intensity plot to {filepath_subplots}")

    if show_figs:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_three_intensities(intensities: Dict, metadata: Dict, show_figs:bool=True, save_figs:bool=False, vert_layout:bool=False, figsize:tuple = (20,7), normalized:bool=False) -> None: 
    # TODO: added normalized parameter. remove normalized metadata from intensity data
    """
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensity values to be plotted 
        metadata (Dict[str, str, int]) - plotting metadata from load_aging_knee_coords()
        show_figs (bool) - show or hide figs. Default is True
        save_figs (bool) - save or don't safe figs to standard "figures/" directory. Default is False
    """
    if VERBOSE: print("plot_three_intensities() called!")

    normalized = intensities["normalized"] # Get intensity metadata (boolean flag)
    keys = ["l", "m", "r"] # Hardcoded 
    plt.style.use('default')

    # Prepare formatting strings
    if normalized: ttl_sfx = "(Normalized " + metadata["knee_id"] + ")"
    else: ttl_sfx = "(Raw " + metadata["knee_id"] + ")"    
    ttl_pfx = {"l": "Left", "m": "Middle", "r": "Right"}
    clrs = {"l": "r", "m": "g", "r": "b"}
    if normalized: sv_fl_pfx = "normalized"
    else: sv_fl_pfx = "raw"

    # Plot three (or more) figs separately
    if vert_layout:
        fig, axes = plt.subplots(len(keys), 1, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, len(keys), figsize=figsize)

    i = 0
    for k in keys:

        # Plot intensities
        f0 = metadata["f0"]
        fN = metadata["fN"]
        frmns = np.arange(f0, fN+1)
        print(k)
        axes[i].plot(frmns, intensities[k][f0:fN+1], color=clrs[k], label=f"{ttl_pfx[k]} knee")

        # Formatting
        axes[i].axvline(metadata["flx_ext_pt"], color="k", linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
        axes[i].legend()

        i+=1

    # Vertical layout formatting
    if vert_layout: 
        axes[0].set_title(f"{metadata['knee_id']} knee pixel intensities")
    else:
        axes[1].set_title(f"Knee pixel intensities {ttl_sfx}")

        # i=0
        # for k in keys:
        #     axes[i].set_title(ttl_pfx[k] + " knee pixel intensities " + ttl_sfx)
        #     i+=1


    if save_figs:
        fn = f"../figures/intensity_plots/{sv_fl_pfx}_separate_{metadata['knee_id']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    # plt.show()


    # Plot three (or more) figs combined
    plt.figure(figsize=figsize)
    for k in keys:
        plt.plot(frmns, intensities[k][f0:fN+1], color=clrs[k], label=ttl_pfx[k] + " knee")
    plt.axvline(metadata["flx_ext_pt"], color='k', linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
    plt.title("Knee pixel intensities " + ttl_sfx)
    plt.legend()

    if save_figs:
        fn = f"../figures/intensity_plots/{sv_fl_pfx}_combined_{metadata['knee_id']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    if show_figs:
        plt.tight_layout()
        plt.show()

    plt.close('all')

    return

def plot_three_derivs(derivs:Dict[str, np.ndarray], metadata:Dict, show_figs=True, save_figs=False, figsize:Tuple[int,int]=(17,10)) -> None:
    """Plots the three derivatives"""
    if VERBOSE: print("plot_three_derivs() called!")

    plt.style.use('default')

    # Prepare formatting strings
    keys = ["l","m","r"]
    colors = {"l":"r","m":"g","r":"b"}
    
    fig, ax = plt.subplots(3,1, figsize=figsize)
    for i in range(3):
        k = keys[i]
        fns = np.arange(metadata["f0"], metadata["f0"] + len(derivs[k]))
        ax[i].plot(fns, derivs[k], label=f"{k} knee", color=colors[k])
        ax[i].axhline(0, color='k', alpha=0.7)
        ax[i].axvline(metadata["flx_ext_pt"], color='k', linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
        ax[i].legend()
    ax[0].set_title(f"Change in intensity per frame ({metadata['knee_id']})")
    ax[-1].set_xlabel(f"Frame number ({metadata['f0']}-{metadata['fN']})")

    if save_figs:
        fn = f"../figures/derivative_plots/derivs_{metadata['knee_id']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    if show_figs: 
        plt.tight_layout()
        plt.show()

    plt.close("all")
    return

# TODO: implement arrow keys traversal
# def show_frames(video:np.ndarray) -> None:
#     """Shows all frames. Press any button to advance, or 'q' to exit"""
#     if VERBOSE: print("show_frames() called!")

#     video = video.copy() # don't modify original
#     h,w = video.shape[1:]
#     btm_l_pos = (10, h - 10)

#     for cf, frame in enumerate(video):
#         cv2.putText(frame, str(cf), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
#                     fontScale = 0.7, color = (255, 255, 255), thickness = 1, lineType=cv2.LINE_AA)
#         cv2.imshow("show_frames()", frame)
#         if cv2.waitKey(0) == ord('q'): break
#     cv2.destroyAllWindows()

#     return
# 
def show_frames(video:np.ndarray, title:str=None, show_num:bool=True) -> None:
    """Shows all frames. Use keys {a,s} to navigate, or 'q' to exit. Accepts a list of videos as input for horizontal concatenation"""
    if VERBOSE: print("show_frames() called!")

    if isinstance(video, List): # TODO: Dynamically accept a list of videos and automatically display them side by side 
        video = np.concatenate(video, axis=2)

    if video.dtype != np.uint8:
        print("show_frames() runtime warning: input video is not np.uint8! Converting...")
        video = np.asarray(video, dtype=np.uint8)

    video = video.copy() # don't modify original
    nfs, h,w = video.shape
    btm_l_pos = (10, h - 10)

    if title is None: title = "show_frames()"

    # Skip through frames with number row
    itvs = np.linspace(0, nfs, 11, dtype=int)[:-1]
    idxs = [ord(str(n)) for n in [1,2,3,4,5,6,7,8,9,0]]
    fn_slcs = dict(zip(idxs, itvs))

    cf=0
    while True:
        frame = video[cf]

        if show_num:
            cv2.rectangle(frame, (0, h-32), (59, h), color=0, thickness=-1)
            cv2.putText(frame, str(cf), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = 255, thickness = 1, lineType=cv2.LINE_AA)
        
        cv2.imshow(title, frame)

        # Controls 
        k = cv2.waitKey(0)
        if k == ord('q'): break
        if k == ord("a"): cf-=1
        if k == ord("s"): cf+=1
        if k in fn_slcs: cf = fn_slcs[k]

        # Edge handling
        cf = cf%nfs # mod num_frames
    cv2.destroyAllWindows()

    return

def show_regions(regions:Dict[str, np.ndarray], keys:List[str]) -> None:
    if VERBOSE: print("show_regions() called!")

    regions = regions.copy()
    n_frames, _, _ = regions[keys[0]].shape # Assume each np.ndarray in regions[] has the same dimensions

    cf = 0
    # for cf in np.arange(n_frames):
    while True:

        f_stack = np.hstack(tuple([utils.crop_frame_square(regions[k][cf], h=350) for k in keys]))
        # f_stack = regions["l"][cf] | regions['m'][cf] | regions['r'][cf]
        h = f_stack.shape[0]
        btm_l_pos = (10, h - 10)

        cv2.putText(f_stack, str(cf), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 0.7, color = (255, 255, 255), thickness = 1, lineType=cv2.LINE_AA)
        

        cv2.imshow(f"show_regions()", f_stack)

        # Controls
        c = cv2.waitKey(0)
        if c == ord('q'): break
        if c == ord("a"): cf-=1
        if c == ord("d"): cf+=1
        cf = cf%n_frames # Wrap frames

    cv2.destroyAllWindows()

def _draw_mask_outline(frame:np.ndarray, mask_segment:np.ndarray) -> np.ndarray:
    """Helper function. Draws the outline of a binary mask on the frame and returns it"""

    assert frame.shape == mask_segment.shape

    frame = frame.copy()
    mask_segment = mask_segment.copy()

    contours, _ = cv2.findContours(mask_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (127,0,0), 1)

    return frame

def draw_mask_boundary(video:np.ndarray, mask:np.ndarray, show_video:bool=True) -> np.ndarray:
    """Draws the boundary of a binary mask on each frame of the video."""
    if VERBOSE: print("draw_mask_boundary() called!")

    assert video.shape == mask.shape

    video = video.copy()
    nfs,h,w = video.shape

    for cf in range(nfs):
        frame = video[cf]
        mask_f = mask[cf]

        video[cf] = _draw_mask_outline(frame, mask_f)

    if show_video: show_frames(video)

    return video




def draw_current_frame_num(frame:np.ndarray, frame_num:int) -> None:
    """Draws current frame number on the bottom left corner, directly on the frame. Modifies input frame."""

    h, w = frame.shape
    btm_l_pos = (10, h - 10)
    
    # Draw line
    cv2.putText(frame, str(frame_num), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale = 0.7, color = (255, 255, 255), thickness = 1, lineType=cv2.LINE_AA)
    
    return


def draw_radial_masks(video:np.ndarray, radial_masks:np.ndarray, show_video:bool=True, frame_offset:int=0) -> None:
    if VERBOSE: print("draw_radial_masks() called!")

    radial_masks = np.asarray(radial_masks)
    video = video.copy()

    nsegs, nfrs, h, w = radial_masks.shape

    for cf in range(nfrs):

        frame = video[cf]

        for seg in range(nsegs):
            msk = (radial_masks[seg, cf] > 0).astype(np.uint8) * 255 # Cast to 255 intensity uint8
            video[cf] = _draw_mask_outline(frame, msk)

        draw_current_frame_num(frame, cf+frame_offset)

    if show_video: show_frames(video, show_num=False)

    return video


def draw_middle_lines(video:np.ndarray, show_video:bool=True, hplace:float=0.5, vplace:float=0.5) -> np.ndarray:
    """Draws lines through the middle of the frame. Lines can be offset by optional params"""
    if VERBOSE: print("draw_middle_lines() called!")

    if not 0 <= hplace <= 1 and not 0 <= vplace <= 1:
        raise ValueError("draw_middle_lines(): optional params need to be in [0,1]")

    video = video.copy()
    h, w = video.shape[1:]  
    for cf, frame in enumerate(video):
        cv2.line(frame, (0, round(h*hplace)), (w, round(h*hplace)), (255, 255, 255), 1)  # Horizontal line
        cv2.line(frame, (round(w*vplace), 0), (round(w*vplace), h), (255, 255, 255), 1)  # Vertical line
    
    if show_video: show_frames(video)

    return video

def draw_point(video:np.ndarray, pt:Tuple[int,int], show_video:bool=True) -> np.ndarray:
    """Draws a point (x,y) on the frame and displays it"""
    if VERBOSE: print("draw_point() called!")

    video = video.copy()
    for cf, frame in enumerate(video):
        cv2.circle(frame, pt[cf], 3, (255,255,255), -1)

    if show_video: show_frames(video)

    return video

def _draw_points(frame:np.ndarray, pts:np.ndarray) -> np.ndarray:
    """Helper func to draw_points(). Draws a set of points in pts directly on the frame."""
    pts = np.array(pts)
    if pts.ndim != 2 or pts.shape[1] != 2: 
        raise ValueError(f"_draw_points(): 'pts' must be 2D array with shape (N, 2). Shape {pts.shape} was given")
    
    for pt in pts:
        cv2.circle(frame, tuple(pt), 1, (255,255,255))
    
def draw_points(video:np.ndarray, pts:np.ndarray, show_video:bool=True) -> np.ndarray:
    """For every frame, draws a set of points [[x1,y1], [x2,y2], ..., [xn,yn]] and displays it"""
    if VERBOSE: print("draw_points() called!")

    if video.shape[0] != pts.shape[0]:
        raise ValueError("draw_points(): video and pts arrays must have same number of rows")

    video = video.copy() # for safety
    pts = pts.copy()

    for cf in range(video.shape[0]):
        cpts = np.asarray(pts[cf])
        frame = np.asarray(video[cf])
        # print(cpts) # debug
        # print(cpts.size) # debug
        if cpts.size == 0: continue
        _draw_points(frame, cpts)

    if show_video: show_frames(video)
    
    return video

def draw_line(video:np.ndarray, pt1:List[Tuple[int,int]], pt2:List[Tuple[int,int]], show_video:bool=True) -> np.ndarray:
    """Draws a line on a video between two points"""
    if VERBOSE: print("draw_line() called!")

    video = video.copy()
    for cf, frame in enumerate(video):
        cv2.line(frame, pt1[cf], pt2[cf], (255,255,255), 1)
    
    if show_video: show_frames(video)
    
    return video

def _draw_numbers_on_circle_coords(frame:np.ndarray, coords:np.ndarray):
    """Helper function. Draws number in each slice."""

    h, w = frame.shape
    N, _ = coords.shape

    cx, cy = w / 2, h / 2  # note (x, y) order for center
    theta = np.pi / N

    rot_mx = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    for n in range(N):

        x,y = coords[n]
        vec = np.array([x - cx, y - cy]) # vector from center
        rot_vec = rot_mx @ vec 
        nx, ny = rot_vec + np.array([cx, cy]) # translate back
        nx, ny = int(nx), int(ny)

        cv2.putText(frame, str(n), (nx,ny), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 0.3, color = (255,255,0), thickness = 1, lineType=cv2.LINE_AA)

    return

def draw_radial_slice_numbers(video:np.ndarray, coords_on_circle:np.ndarray, show_video:bool=True) -> np.ndarray:
    """Draws the slice number on every frame"""

    nfrs, h, w = video.shape

    for cf in range(nfrs):
        
        frame = video[cf]
        _draw_numbers_on_circle_coords(frame, coords_on_circle[cf])

    if show_video: show_frames(video)

    return video

def rescale_video(video:np.ndarray, scale_factor:int, show_video:bool=True, show_num:bool=False) -> np.ndarray:

    video = video.copy()
    nfrs, h,w = video.shape

    # nh, nw = h*scale_factor, w*scale_factor
    # nh, nw = int(nh), int(nw)

    video_rscld = []
    for cf in range(nfrs):
        frame = cv2.resize(video[cf], None, fx = scale_factor, fy = scale_factor, interpolation=cv2.INTER_LINEAR)
        video_rscld.append(frame)
    video_rscld = np.array(video_rscld)

    if show_video: show_frames(video_rscld, show_num=show_num)

    return video_rscld

def draw_text(frame:np.ndarray, text:str, pos:str='bl') -> np.ndarray:
    """Draws text on a frame. Some planned options in the future"""

    # frame = frame.copy()
    h, w = frame.shape

    positions = {"bl": (20, h-10)} # TODO: other positions
    coords = positions[pos]

    cv2.putText(frame, text, coords, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 0.3, color = (255,255,0), thickness = 1, lineType=cv2.LINE_AA)


    return frame