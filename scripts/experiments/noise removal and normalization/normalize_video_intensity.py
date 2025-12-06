import utils.views as views
import utils.io as io
import utils.utils as utils
import numpy as np
import core.knee_segmentation as ks
import core.radial_segmentation as rdl


video1358 = io.load_nparray(r"..\data\processed\aging_1358_radial_video_N64.npy")#; views.show_frames(video1358, "1358")
video1342 = io.load_nparray(r"..\data\processed\aging_1342_radial_video_N64.npy")#; views.show_frames(video1342*1.5, "1342")
video1339 = io.load_nparray(r"..\data\processed\aging_1339_radial_video_N64.npy")#; views.show_frames(video1339, "1339")


# Video intensity normalization procedure
test_1342 = utils.normalize_video_intensity(video1342)

views.show_frames(test_1342)
