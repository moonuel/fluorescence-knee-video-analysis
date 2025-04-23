import imageio.v2 as imageio
import numpy as np
import cv2

def import_spe(file_path):
    '''
    Description:
        Reads SPE files
    
    Inputs:
        file_path - str: location of the SPE file

    Returns:
        data - NumPy array (width, height, num_frames)
    '''

    reader = imageio.get_reader(file_path, format="SPE")

    num_frames = reader.get_length()
    first_frame = reader.get_data(0)
    height = first_frame.shape[0]
    width = first_frame.shape[1]

    data = np.zeros((height, width, num_frames), dtype = first_frame.dtype)

    for frame in range(num_frames):
        data[:,:, frame] = reader.get_data(frame) 

    return data

def import_tif(file_path):
    '''
    Description:
        Reads TIFF files
    
    Inputs:
        file_path - str: location of the TIFF file

    Returns:
        im - NumPy array (width, height, num_frames)
    '''

    im = imageio.imread(file_path)

    im = np.array(im)
    im = np.moveaxis(im, [0], [2])

    return im

def import_avi(file_path):
    '''
    SLOW
    Description:
        Reads AVI files
    
    Inputs:
        file_path - str: location of the AVI file

    Returns:
        im - NumPy array (width, height, num_frames)
    '''
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    data = []

    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data.append(frame)
    
    cap.release()
    data = np.moveaxis(data, [0], [3])

    return np.array(data)