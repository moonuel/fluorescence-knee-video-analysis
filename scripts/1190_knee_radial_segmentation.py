from utils import utils, io, views
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
import numpy as np




def main():

    video = io.load_nparray("../data/processed/1190_knee_frames_ctrd.npy")
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)

    views.show_frames(video)

    # v1 = views.rescale_video(video, 0.5, False)
    # views.show_frames(v1)


if __name__ == "__main__":
    main()