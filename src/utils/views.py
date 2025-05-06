import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.config import VERBOSE
from typing import Dict, List, Tuple
from src.utils import utils


def plot_coords(video:np.ndarray, coords:pd.DataFrame) -> None:
    """Plots the set of coordinates for the three part segmentation"""
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

def show_frames(video:np.ndarray) -> None:
    """Shows all frames. Press any button to advance, or 'q' to exit"""
    if VERBOSE: print("show_frames() called!")

    video = video.copy() # don't modify original
    h,w = video.shape[1:]
    btm_l_pos = (10, h - 10)

    for cf, frame in enumerate(video):
        cv2.putText(frame, str(cf), btm_l_pos, fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 0.7, color = (255, 255, 255), thickness = 1, lineType=cv2.LINE_AA)
        cv2.imshow("show_frames()", frame)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

    return

def plot_three_intensities(intensities: Dict, metadata: Dict, show_figs:bool=True, save_figs:bool=False, vert_layout:bool=False, figsize:tuple = (20,7)) -> None:
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
        fns = np.arange(metadata["f0"], metadata["f0"] + len(intensities[k]))
        axes[i].plot(fns, intensities[k], color=clrs[k], label=f"{ttl_pfx[k]} knee")

        # Formatting
        axes[i].axvline(metadata["flx_ext_pt"], color="k", linestyle="--", label=f"Start of extension (frame {metadata['flx_ext_pt']})")
        axes[i].legend()

        i+=1

    # Vertical layout formatting
    if vert_layout: 
        axes[0].set_title(f"{metadata['knee_id']} knee pixel intensities")
    else:
        for k in keys:
            axes[i].set_title(ttl_pfx[k] + " knee pixel intensities " + ttl_sfx)


    if save_figs:
        fn = f"../figures/intensity_plots/{sv_fl_pfx}_separate_{metadata['knee_id']}.png"
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

def display_regions(regions:Dict[str, np.ndarray], keys:List[str]) -> None:
    if VERBOSE: print("display_regions() called!")

    regions = regions.copy()
    n_frames = regions[keys[0]].shape[0] # Assume each np.ndarray in regions[] has the same dimensions
    for cf in np.arange(n_frames):
        f_stack = np.hstack(tuple([utils.crop_frame_square(regions[k][cf], h=350) for k in keys]))
        cv2.imshow(f"display_regions()", f_stack)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

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

def draw_points_(frame:np.ndarray, pts:np.ndarray) -> np.ndarray:
    """Helper func to draw_points(). Draws a set of points in pts directly on the frame."""
    if pts.ndim != 2 or pts.shape[1] != 2: 
        raise ValueError(f"draw_points_(): 'pts' must be 2D array with shape (N, 2)")
    
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
        draw_points_(video[cf], pts[cf])

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