import cv2
import numpy as np

def moving_average_curve(curve, window_size=30):
    """Applies a moving average filter to smooth the trajectory."""
    filtered_curve = np.copy(curve)
    for i in range(len(curve)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(curve), i + window_size // 2)
        filtered_curve[i] = np.mean(curve[start_idx:end_idx], axis=0)
    return filtered_curve

def stabilize_video(input_path, output_path, smoothing_window=30):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the output video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=False)

    # Read the first frame and convert to grayscale
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track in the first frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    # Store transformation history
    transforms = np.zeros((frame_count - 1, 3), np.float32)

    # Process each frame
    for i in range(1, frame_count):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow to track points
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        valid_prev_pts = prev_pts[status == 1]
        valid_curr_pts = curr_pts[status == 1]

        # Estimate transformation (rigid transform)
        transform_matrix, _ = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)
        
        # Extract translation and rotation from the transformation matrix
        dx = transform_matrix[0, 2]
        dy = transform_matrix[1, 2]
        da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        
        # Store the transformation
        transforms[i - 1] = [dx, dy, da]

        # Update previous frame and points
        prev_gray = curr_gray
        prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    # Compute the cumulative sum to get the trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth the trajectory using a moving average filter
    smoothed_trajectory = moving_average_curve(trajectory, window_size=smoothing_window)

    # Calculate the difference between the smoothed trajectory and the original
    difference = smoothed_trajectory - trajectory
    smooth_transforms = transforms + difference

    # Reset the video capture to stabilize the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Stabilize each frame using the calculated transforms
    for i in range(frame_count - 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the smoothed transformation
        dx, dy, da = smooth_transforms[i]

        # Build the transformation matrix
        transform_matrix = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        # Apply the affine transformation to stabilize the frame
        stabilized_frame = cv2.warpAffine(gray_frame, transform_matrix, (frame_width, frame_height))

        # Fix border artifacts
        center = (frame_width // 2, frame_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, da * 180 / np.pi, 1.0)
        stabilized_frame = cv2.warpAffine(stabilized_frame, rotation_matrix, (frame_width, frame_height))

        # Write the stabilized frame to the output video
        out.write(stabilized_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Stabilized video saved as {output_path}")

# Run the stabilization function
stabilize_video('video_1.avi', 'video_1_stabilized_python.avi')

