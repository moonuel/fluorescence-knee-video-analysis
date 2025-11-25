import src.utils.views as views
import src.utils.io as io

video1358 = io.load_nparray(r"..\data\processed\1358_aging_radial_video_N64.npy")
video1342 = io.load_nparray(r"..\data\processed\1342_aging_radial_video_N64.npy")
video1339 = io.load_nparray(r"..\data\processed\1339_aging_radial_video_N64.npy")

views.show_frames([video1358, video1339, video1342], "1358 vs 1339 and 1342")