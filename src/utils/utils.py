import os
from datetime import datetime
import cv2
import numpy as np
from typing import Tuple, List
import math
from config import VERBOSE
import numpy as np
import multiprocessing as mp
import math
import logging
from typing import Callable, Optional

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


def crop_video_square(video: np.ndarray, h: int, w: int = None) -> np.ndarray:
    """
    Crops the input video into a centered rectangle of dimensions h-by-w pixels
    (default is h-by-h square).
    
    Inputs:
        video (np.ndarray): video array with dimensions (nframes, H, W)
        h (int): target height
        w (int): target width (defaults to h)
    Outputs:
        np.ndarray: cropped video with shape (nframes, h, w)
    """
    if w is None:
        w = h

    # original spatial dimensions
    H, W = video.shape[1], video.shape[2]

    # calculate crop indices (centered)
    top = (H - h) // 2
    left = (W - w) // 2

    # crop in one go (broadcasts across frames)
    return video[:, top:top+h, left:left+w]


def pixels_left_of_line(frame, p1, p2):
    """Classify pixels as left or right of the line (p1, p2) using the cross product."""
    H, W = frame.shape
    
    # Generate pixel coordinates
    y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Compute the signed area (cross product)
    cross = (x_indices - p1[0]) * (p2[1] - p1[1]) - (y_indices - p1[1]) * (p2[0] - p1[0])
    
    # Classification: left side is positive, right side is negative
    return ((cross > 0) * 255).astype(np.uint8)  # Boolean mask: True if left, False if right



def blur_video(video:np.ndarray, kernel_dims:Tuple[int,int]=(25,25), sigma:int=0) -> np.ndarray:
    """Implements a Gaussian blur over a grayscale video with dimensions (nframes,hgt,wth)"""
    if VERBOSE: print("blur_video() called!")

    video_b = []
    for _, frame in enumerate(video):
        frame = cv2.GaussianBlur(frame, kernel_dims, sigma)
        video_b.append(frame)
    video_b = np.array(video_b)

    return video_b


def morph_open(video:np.ndarray, kernel_size:Tuple[int,int]) -> np.ndarray:
    "Implements a morphological opening operation over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("morph_open() called!")

    video = np.asarray(video)
    dtype = video.dtype

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    video_o = []
    for _, frame in enumerate(video):
        if dtype != np.uint8: frame = frame.astype(np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        video_o.append(frame.astype(dtype))
    video_o = np.array(video_o, dtype=dtype)

    return video_o

def morph_close(video:np.ndarray, kernel_size:Tuple[int,int]) -> np.ndarray:
    "Implements a morphological closing operation over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("morph_close() called!")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    video_c = []
    for _, frame in enumerate(video):
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        video_c.append(frame)
    video_c = np.array(video_c)

    return video_c

def morph_erode(video:np.ndarray, kernel_size:Tuple[int,int]) -> np.ndarray:
    "Implements a morphological erosion operation over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("morph_erode() called!")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    video_c = []
    for _, frame in enumerate(video):
        frame = cv2.erode(frame, kernel)
        video_c.append(frame)
    video_c = np.array(video_c)

    return video_c

def morph_dilate(video:np.ndarray, kernel_size:Tuple[int,int]) -> np.ndarray:
    "Implements a morphological dilation operation over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("morph_dilate() called!")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    video_c = []
    for _, frame in enumerate(video):
        frame = cv2.dilate(frame, kernel)
        video_c.append(frame)
    video_c = np.array(video_c)

    return video_c

def rotate_video(video:np.ndarray, angle:int, center:Tuple[int,int]=None) -> np.ndarray:
    "Rotates (cw) an entire video (nframes,hgt,wth) around the center by a given angle. Optionally specify the center of rotation"
    if VERBOSE: print("rotate_video() called!")

    h,w = video.shape[1:]
    if center is None:
        center = (w//2, h//2) # Default to frame center

    rot_mx = cv2.getRotationMatrix2D(center, -angle, 1)
    video_r = np.array([
        cv2.warpAffine(frame, rot_mx, (w, h))
        for frame in video
    ])

    return video_r
    

def log_transform_video(video:np.ndarray, gain:float=1.0) -> np.ndarray:
    """Log transforms every frame in a video"""

    video = video.copy()
    for cf, frame in enumerate(video):
        # video[cf] = _log_transform_frame(frame, gain)
        # video[cf] = _adjust_log_cv(frame, gain)
        video[cf] = _log_transform_opencv(frame, gain=1)

    return video

def _log_transform_frame(frame:np.ndarray, gain:float) -> np.ndarray:
    """Helper function. Log transforms a frame"""
    # Convert to float32 for precision and avoid overflow
    flt_frm = frame.astype(np.float32)

    # Apply log transform
    log_frame = gain * np.log1p(flt_frm)  # log(1 + I)

    # Normalize to [0,255] and convert back to uint8
    log_frame = cv2.normalize(log_frame, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(log_frame)

def _adjust_log_cv(img, gain=1.0):
    """Helper function. Log transforms a frame. Similar to skimage.exposure.adjust_log()?"""
    img = img.astype(np.float32)
    img /= img.max()  # Normalize to [0, 1]
    log_img = gain * np.log1p(img)  # log(1 + I)
    log_img = np.clip(log_img / log_img.max(), 0, 1)  # Normalize log output
    return np.uint8(log_img * 255)  # Scale to [0, 255]

def _log_transform_opencv(frame: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Log or inverse log correction using OpenCV-compatible operations."""
    if np.any(frame < 0):
        raise ValueError("Image contains negative values. Log transform is undefined.")

    frame = frame.astype(np.float32)  # work in float
    scale = 255.0 

    frame = frame / scale  # normalize to [0,1]

    out = np.log2(1 + frame) * scale * gain

    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)  # convert back for OpenCV display

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Callable, Optional

def split_into_batches(video: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Split video into batches of frames."""
    return [video[i:i+batch_size] for i in range(0, len(video), batch_size)]

def concatenate_batches(batches: list[np.ndarray]) -> np.ndarray:
    """Recombine processed frame batches into a full video."""
    return np.concatenate(batches, axis=0)

def process_batch(batch: np.ndarray, frame_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply frame_function to each frame in the batch."""
    return np.stack([frame_function(frame) for frame in batch], axis=0)

def process_batch_video(batch: np.ndarray, video_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply video_function to each frame in the batch."""
    return video_function(batch)

def choose_batch_size(n_frames: int, max_batch_size: int = 512) -> int:
    """Determine a batch size based on CPU count and number of frames."""
    num_cores = cpu_count()
    estimated = (n_frames + num_cores - 1) // num_cores  # ceil division
    return min(estimated, max_batch_size)

def parallel_process_frames(
    video: np.ndarray,
    frame_function: Callable[[np.ndarray], np.ndarray],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply a frame-level pure function to a video in parallel, in batches.
    Accepts a frame-wise pure function. 

    Parameters:
        video (np.ndarray): Input video of shape (N, H, W[, C]).
        frame_function (Callable): A pure function that processes a single frame.
        batch_size (int, optional): Number of frames per batch. If None, determined automatically.
        num_workers (int, optional): Number of parallel processes. Defaults to number of CPU cores.

    Returns:
        np.ndarray: Processed video of same shape.

    Example:
        >>> import cv2
        >>> from functools import partial
        >>> def gaussian_blur(frame, ksize):
        ...     return cv2.GaussianBlur(frame, ksize=ksize, sigmaX=0)
        >>> blur_fn = partial(gaussian_blur, ksize=(5, 5))
        >>> output = parallel_process_video(video, blur_fn)
    """

    n_frames = len(video)

    if batch_size is None:
        batch_size = choose_batch_size(n_frames)

    if num_workers is None:
        num_workers = cpu_count()

    batches = split_into_batches(video, batch_size)

    # Bind the frame_function into the batch processor
    batch_processor = partial(process_batch, frame_function=frame_function)

    with Pool(processes=num_workers) as pool:
        results = pool.map(batch_processor, batches)

    return concatenate_batches(results)

def parallel_process_video(
    video: np.ndarray,
    video_function: Callable[[np.ndarray], np.ndarray],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply a batch-wise pure function to a video in parallel, processing in batches.

    This function splits the input video into batches of frames, applies the
    provided batch-wise `video_function` to each batch in parallel using multiple
    worker processes, and then recombines the processed batches into a single video.

    Parameters:
        video (np.ndarray): Input video as a NumPy array of shape (N, H, W[, C]),
            where N is the number of frames, H and W are height and width, and C is optional channels.
        video_function (Callable[[np.ndarray], np.ndarray]): A pure function that
            accepts a batch of frames (shape (batch_size, H, W[, C])) and returns
            a processed batch of the same shape.
        batch_size (int, optional): Number of frames per batch to process in each worker.
            If None, the batch size is determined automatically based on the video length and CPU cores.
        num_workers (int, optional): Number of parallel worker processes to use.
            If None, defaults to the number of CPU cores.

    Returns:
        np.ndarray: The processed video, with the same shape as the input.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from functools import partial
        >>>
        >>> # Example batch-wise function applying Gaussian blur to each frame
        >>> def batch_gaussian_blur(batch: np.ndarray, ksize=(5, 5)) -> np.ndarray:
        ...     # Apply OpenCV GaussianBlur frame-by-frame in the batch
        ...     return np.array([cv2.GaussianBlur(frame, ksize, sigmaX=0) for frame in batch])
        >>>
        >>> # Create a partial with fixed kernel size
        >>> blur_fn = partial(batch_gaussian_blur, ksize=(7, 7))
        >>>
        >>> # Assume `video` is a NumPy array of shape (N, H, W)
        >>> processed_video = parallel_process_video(video, blur_fn, batch_size=64, num_workers=8)
    """
    n_frames = len(video)

    if batch_size is None:
        batch_size = choose_batch_size(n_frames)

    if num_workers is None:
        num_workers = cpu_count()

    batches = split_into_batches(video, batch_size)

    # Bind the video_function into the batch processor
    batch_processor = partial(process_batch_video, video_function=video_function)

    with Pool(processes=num_workers) as pool:
        results = pool.map(batch_processor, batches)

    return concatenate_batches(results)


def choose_batch_size_video(n_frames: int) -> int:
    """Heuristic for selecting a default batch size."""
    return 50 if n_frames >= 200 else max(10, n_frames // 4)

def parallel_process_video(
    video: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_workers: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Apply a batch-wise pure function to a video in parallel.

    This function splits the input video into batches of frames, applies the
    given batch-wise function in parallel using multiple processes, and 
    returns the processed video with the same shape.

    Parameters:
        video (np.ndarray): Input video of shape (N, H, W[, C]), where N is the number of frames.
        func (Callable): A function that accepts a batch of frames 
            (i.e., an array of shape (batch_size, H, W[, C])) and returns a batch of processed frames.
        batch_size (int, optional): Number of frames per batch. If None, chosen automatically.
        num_workers (int, optional): Number of parallel worker processes. If None, chosen automatically.
        max_workers (int, optional): Maximum number of processes allowed (defaults to all available CPU cores).
        verbose (bool, optional): If True, enables logging output.

    Returns:
        np.ndarray: Processed video of the same shape as the input.

    Example:
        >>> import cv2
        >>> from functools import partial
        >>> def batch_blur(batch: np.ndarray, ksize=(5, 5)) -> np.ndarray:
        ...     return np.array([cv2.GaussianBlur(f, ksize, sigmaX=0) for f in batch])
        >>> blur_fn = partial(batch_blur, ksize=(7, 7))
        >>> output = parallel_process_video(video, blur_fn, verbose=True)
    """
    n_frames = len(video)

    # Configure logging
    logger = logging.getLogger("parallel_process_video")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    cpu_limit = mp.cpu_count()
    max_workers = max_workers or cpu_limit

    # Resolve parameter interaction
    if batch_size is None and num_workers is None:
        batch_size = choose_batch_size_video(n_frames)
        num_workers = min(max_workers, math.ceil(n_frames / batch_size))
        logger.info(f"[Auto] batch_size={batch_size}, num_workers={num_workers}")

    elif batch_size is not None and num_workers is None:
        num_workers = min(max_workers, math.ceil(n_frames / batch_size))
        logger.info(f"[Auto] num_workers={num_workers} for batch_size={batch_size}")

    elif batch_size is None and num_workers is not None:
        num_workers = min(num_workers, max_workers)
        batch_size = math.ceil(n_frames / num_workers)
        logger.info(f"[Auto] batch_size={batch_size} for num_workers={num_workers}")

    else:
        num_workers = min(num_workers, max_workers)
        logger.info(f"[Manual] Using batch_size={batch_size}, num_workers={num_workers}")

    # Validate parameters
    assert batch_size > 0 and num_workers > 0, "batch_size and num_workers must be positive integers"

    # Create batches
    batches = [video[i:i + batch_size] for i in range(0, n_frames, batch_size)]
    logger.info(f"Created {len(batches)} batches, each with up to {batch_size} frames")

    # Parallel execution
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(func, batches)

    # Combine results
    return np.concatenate(results, axis=0)

