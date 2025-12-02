(11 July 2025)

These intensity plots and videos were generated to compare the performance of the naive radial segmentation algorithm (assumes femur tip is the center of the video) with the revised one (samples points along the interior of the fluorescence boundary to estimate the position of the femur).

The old and new method were compared using the same preprocessing steps: centroid stabilization, Gaussian blur with a 31*31 kernel and sigma=0, and thresholding region with an Otsu threshold rescaling of 0.6. 