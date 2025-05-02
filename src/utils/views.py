import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.config import VERBOSE
from typing import Dict, List
from src.utils import utils


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

def view_frames(video:np.ndarray) -> None:
    """Shows all frames. Press any button to advance, or 'q' to exit"""
    if VERBOSE: print("view_frames() called!")

    for cf, frame in enumerate(video):
        cv2.imshow("view_frames()", frame)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()

    return

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

def display_regions(regions:Dict[str, np.ndarray], keys:List[str]) -> None:
    if VERBOSE: print("display_regions() called!")

    n_frames = regions[keys[0]].shape[0] # Assume each np.ndarray in regions[] has the same dimensions
    for cf in np.arange(n_frames):
        f_stack = np.hstack(tuple([utils.crop_square_frame(regions[k][cf], n=350) for k in keys]))
        cv2.imshow(f"display_regions()", f_stack)
        if cv2.waitKey(0) == ord('q'): break
    cv2.destroyAllWindows()