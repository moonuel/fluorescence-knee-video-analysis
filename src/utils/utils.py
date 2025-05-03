import os
from datetime import datetime
import cv2
import numpy as np
from typing import Tuple, List
import math
from src.config import VERBOSE

def print_last_modified(filepath):
    """
    Prints the last modified date and time of the specified file.

    Parameters:
    - filepath (str): The path to the file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - PermissionError: If the program does not have permission to access the file.
    - Exception: For any other unexpected errors.
    """
    try:
        # Get the last modified timestamp
        date_modified = os.path.getmtime(filepath)
        
        # Convert the timestamp to a human-readable format
        date_modified = datetime.fromtimestamp(date_modified).strftime('%Y-%m-%d %H:%M:%S')
        
        # Print the result
        print(f"file {filepath} last modified {date_modified}")
    
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied to access the file '{filepath}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function to crop an image with specific margins
def crop_image_with_margins(image, top_margin, left_margin, bottom_margin, right_margin):
    """
    Crop a region from the image by specifying margins.

    Parameters:
    - image: The input image.
    - top_margin: Pixels to remove from the top.
    - left_margin: Pixels to remove from the left.
    - bottom_margin: Pixels to remove from the bottom.
    - right_margin: Pixels to remove from the right.

    Returns:
    - cropped_image: The cropped part of the image.
    """
    # Calculate new coordinates for cropping
    y = top_margin
    x = left_margin
    h = image.shape[0] - top_margin - bottom_margin
    w = image.shape[1] - left_margin - right_margin
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def display_side_by_side(frames: Tuple[np.ndarray, ...], title: str = ""):

    '''
    Display arbitrarily many frames side-by-side for easy comparison.
    
    Parameters:
    - frames (tuple[np.ndarray]): A tuple containing image frames as NumPy arrays.
    - title (str): Optional title for the display window.
    '''
    
    if not frames:
        raise ValueError("The frames tuple must contain at least one image.")

    # Find the maximum height and width among all frames
    max_height = max(frame.shape[0] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)

    # Pad frames to have the same dimensions
    padded_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        top = (max_height - h) // 2
        bottom = max_height - h - top
        left = (max_width - w) // 2
        right = max_width - w - left

        padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_frames.append(padded_frame)

    # Combine frames horizontally
    combined_frame = cv2.hconcat(padded_frames)

    # Resize if too large for display
    if combined_frame.shape[1] > 1920:
        scale_factor = 1920 / combined_frame.shape[1]
        new_size = (1920, int(combined_frame.shape[0] * scale_factor))
        combined_frame = cv2.resize(combined_frame, new_size)

    # Display the combined frame
    # cv2.imshow(title, combined_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return combined_frame, combined_frame.shape


def draw_reference_lines(frame, horizontal_line_placement=0.5, vertical_line_placement=0.5):
    h, w = frame.shape[:2]  
    cv2.line(frame, (0, round(h*horizontal_line_placement)), (w, round(h*horizontal_line_placement)), (255, 255, 255), 1)  # Horizontal line
    cv2.line(frame, (round(w*vertical_line_placement), 0), (round(w*vertical_line_placement), h), (255, 255, 255), 1)  # Vertical line
    return frame

def centroid_stabilization(frame: np.ndarray, blur_strength: int = 41, thresh_scale:int=0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stabilizes a frame by centering the non-zero region in a binary mask.

    Parameters:
        frame (np.ndarray): Input grayscale frame.
        blur_strength (int): Gaussian blur kernel size (must be an odd number).
        thresh_scale (int): To rescale the Otsu threshold value

    Returns:
        centered_frame (np.ndarray): Translated frame with the centroid aligned to the center.
        translation_mx (np.ndarray): Translation matrix used to obtain centered frame. 
                                     Useful for aligning other data
    """

    # Validate blur strength
    if blur_strength % 2 == 0 and blur_strength > 0:
        print(f"centroid_stabilization(): blur_strength={blur_strength} must be a positive int")
        blur_strength = abs(blur_strength)
        blur_strength = blur_strength + 1

    # Convert BGR to grayscale if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame.copy()

    # Apply Gaussian blur to create a smoothed copy
    if blur_strength == 0:
        blurred_frame = frame_gray # Skip the blurring step 
    else:
        blurred_frame = cv2.GaussianBlur(frame_gray, (blur_strength, blur_strength), 0)

    # Apply automatic thresholding to create a binary mask
    thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = int(thresh_scale*thresh_val) # manually adjust the threshold value to lower it if needed
    _, binary_mask = cv2.threshold(blurred_frame, thresh_val, 255, cv2.THRESH_BINARY)


    # Compute the centroid of the binary mask
    M = cv2.moments(binary_mask)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
    else:
        # Default to center if no mass is detected
        centroid_x, centroid_y = frame.shape[1] // 2, frame.shape[0] // 2  

    # Compute translation to center the object
    offset_x = (frame.shape[1] // 2) - centroid_x
    offset_y = (frame.shape[0] // 2) - centroid_y

    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

    # Apply the translation to the original (not blurred) frame
    centered_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))

    return centered_frame, translation_matrix

def draw_scale_bar(frame: np.ndarray, pixel_scale: float, rescale_factor: float, bar_height=7, color=(255, 0, 0), thickness=1, line_type=cv2.LINE_8):
    """
    Draws a 1 cm scale bar on the frame.

    Args:
        frame: The image/frame on which to draw the scale bar.
        pixel_scale: The scale factor in pixels per mm.
        bar_height: The height of the endpoint bars (default is 7 pixels).
        color: The color of the scale bar (default is red in BGR format).
        thickness: The thickness of the lines (default is 1).
        line_type: The type of line (default is cv2.LINE_8).
    """



    # rescale_factor = int(round(rescale_factor))

    # Calculate the length of 1 cm in pixels
    one_cm = round(pixel_scale * rescale_factor * 10)

    # Define the start and end coordinates of the scale bar
    h, w = frame.shape[:2]  # Get the height and width of the frame
    start_of_scale_coords = ((w - one_cm)//2, h - 20)  # Start position of the scale bar
    end_of_scale_coords = (start_of_scale_coords[0] + one_cm, h - 20)  # End position of the scale bar

    # Draw the 1 cm scale bar
    cv2.line(frame, start_of_scale_coords, end_of_scale_coords, color=color, thickness=thickness, lineType=line_type)

    # Calculate shifted coordinates for the endpoint bars
    start_plus_height = (start_of_scale_coords[0], start_of_scale_coords[1] + bar_height)
    start_minus_height = (start_of_scale_coords[0], start_of_scale_coords[1] - bar_height)
    end_plus_height = (end_of_scale_coords[0], end_of_scale_coords[1] + bar_height)
    end_minus_height = (end_of_scale_coords[0], end_of_scale_coords[1] - bar_height)

    # Draw the endpoint bars
    cv2.line(frame, start_plus_height, start_minus_height, color=color, thickness=thickness, lineType=line_type)
    cv2.line(frame, end_plus_height, end_minus_height, color=color, thickness=thickness, lineType=line_type)


def rescale_frame(frame, pixel_scale, scale_factor=1):
    """
    Rescales the frame such that 100 pixels = 1 cm, adjusted by the scale_factor.

    Args:
        frame (np.ndarray): The input frame/image to rescale.
        pixel_scale (float): The current scale of the frame in pixels/mm.
        scale_factor (float): The factor by which to adjust the scale. scale_factor = 0 bypasses the function. 
                             For example, scale_factor = 3 means 300 pixels = 1 cm.

    Returns:
        np.ndarray: The rescaled frame.
        float: The rescale_factor, which describes how much the frame was resized.
                          rescale_factor > 1 means the frame was enlarged, and rescale_factor < 1 means the frame was shrunk.
    """

    # Bypass the function if scale_factor is 0. 
    if scale_factor == 0:
        return frame, 1

    # Calculate the desired pixel density (pixels per mm) 
    # Default: 100 pixels = 1 cm => 10 pixels = 1 mm
    desired_pixel_scale = 10 * scale_factor 

    # Compute the scaling factor for resizing
    rescale_factor = desired_pixel_scale / pixel_scale  

    # Resize the frame using the scaling factor
    new_width = int(frame.shape[1] * rescale_factor)
    new_height = int(frame.shape[0] * rescale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_frame, rescale_factor


import numpy as np

def find_closest_point(mask: np.ndarray, axis: int = 0):
    """
    Finds the closest point in a binary mask to one side of the frame.

    Parameters:
        mask (np.ndarray): A binary image mask.
        axis (int): The side of the frame we want the point closest to.
            axis = 0   -> closest to bottom (max y) (Default)
            axis = 1   -> closest to left (min x)
            axis = 2   -> closest to top (min y)
            axis = 3   -> closest to right (max x)

    Returns:
        (int, int): A tuple (x, y) returning the coordinates of the desired point.
    """

    y, x = np.nonzero(mask)
    if y.size == 0:
        return None  # No points found

    # Normalize axis to be in range {0, 1, 2, 3}
    axis %= 4  

    # Dictionary mapping axis values to corresponding functions
    index_selector = {
        0: np.argmax(y),  # Bottom (max y)
        1: np.argmin(x),  # Left (min x)
        2: np.argmin(y),  # Top (min y)
        3: np.argmax(x)   # Right (max x)
    }

    idx = index_selector[axis]
    return x[idx], y[idx]

import numpy as np

import numpy as np

def crop_frame_square(frame: np.ndarray, h: int, w: int = None) -> np.ndarray:
    """
    Crops the input frame into a square or rectangular shape around the center.

    Inputs:
        frame (np.ndarray): The input frame to be cropped.
        h (int): The desired height of the cropped frame.
        w (optional; int): The desired width of the cropped frame. If None, a square crop (h x h) is used.

    Returns:
        np.ndarray: The cropped frame.
    """
    he, we = frame.shape[:2]

    if w is None:
        w = h # default to square crop

    center_y, center_x = he // 2, we // 2

    half_h, half_w = h // 2, w // 2

    y1 = max(center_y - half_h, 0)
    y2 = min(center_y + half_h + (h % 2), he)

    x1 = max(center_x - half_w, 0)
    x2 = min(center_x + half_w + (w % 2), we)

    return frame[y1:y2, x1:x2]


def crop_video_square(video:np.ndarray, h:int, w:int=None) -> np.ndarray:
    """Crops the input video into a centered square of dimensions h-by-h pixels (optionally, to h-by-w pixels)
    Inputs: 
        video (np.ndarray): video array with dimensions (nframes,h,w)
    Outputs:
        video (np.ndarray): same video crame but cropped around the centre
    """
    if VERBOSE: print("crop_video_square() called!")
    
    if w is None:
        w = h # default to square frame

    vid_c = []
    for _, frame in enumerate(video):
        f_c = crop_frame_square(frame, h, w)
        vid_c.append(f_c)
    vid_c = np.array(vid_c)
    print(video.shape)
    print(vid_c.shape)
    return vid_c

def pixels_left_of_line(frame, p1, p2):
    """Classify pixels as left or right of the line (p1, p2) using the cross product."""
    H, W = frame.shape
    
    # Generate pixel coordinates
    y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Compute the signed area (cross product)
    cross = (x_indices - p1[0]) * (p2[1] - p1[1]) - (y_indices - p1[1]) * (p2[0] - p1[0])
    
    # Classification: left side is positive, right side is negative
    return ((cross > 0) * 255).astype(np.uint8)  # Boolean mask: True if left, False if right

from typing import List, Tuple
import numpy as np

def get_N_points_on_circle(ctr_pt: Tuple[int, int], ref_pt: Tuple[int, int], N: int) -> List[Tuple[int, int]]:
    """Returns a list of N equally spaced points on a circle, arranged clockwise.
    
    The circle is defined by:
    - Center point `ctr_pt`
    - Radius = distance between `ctr_pt` and `ref_pt`
    - First point is `ref_pt`, followed by the remaining points in clockwise order.

    Args:
        ctr_pt: (x, y) center of the circle.
        ref_pt: (x, y) reference point on the circle (first point in the output).
        N: Number of points to generate. If N=1, returns [ref_pt].

    Returns:
        List of (x, y) tuples representing the N points on the circle.
    """

    cx, cy = ctr_pt
    rx, ry = ref_pt
    radius = math.hypot(rx - cx, ry - cy)
    start_angle = math.atan2(ry - cy, rx - cx)  # Angle of ref_pt
    
    points = []
    for i in range(N):
        angle = start_angle - 2 * math.pi * i / N  # Clockwise
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((round(x), round(y)))
    
    return points

def blur_video(video:np.ndarray, kernel_dims:Tuple[int,int], sigma:int=0) -> np.ndarray:
    """Implements a Gaussian blur over a grayscale video with dimensions (nframes,hgt,wth)"""
    if VERBOSE: print("blur_video() called!")

    video_b = []
    for _, frame in enumerate(video):
        frame = cv2.GaussianBlur(frame, kernel_dims, sigma)
        video_b.append(frame)
    video_b = np.array(video_b)

    return video_b

def mask_adaptive(video:np.ndarray, block_size:int, adj_value:int) -> np.ndarray:
    "Implements an adaptive thresholding mask over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("mask_adaptive() called!")

    masks = []
    for _, frame in enumerate(video):
        mask = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,block_size, adj_value)
        masks.append(mask)
    masks = np.array(masks)
    
    return masks
