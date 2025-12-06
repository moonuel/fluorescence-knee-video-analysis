from utils import io, utils, views
from core import knee_segmentation as ks
from core import data_processing as dp

def main():

    video = io.load_avi("../data/video_1.avi")
    video, translation_mxs = ks.centre_video(video)

    io.save_nparray(video, "../data/segmented/normal_knee_processed.npy")
    io.save_nparray(translation_mxs, "../data/segmented/normal_translation_mxs.npy")



    # video = io.load_nparray("../data/segmented/aging_knee_processed.npy")
    # translation_mxs = io.load_nparray("../data/segmented/aging_translation_mxs.npy")


if __name__ == "__main__":
    main()