import utils.io as io
import utils.views as views

def main():

    video = io.load_hdf5_video_chunk("../data/raw/right-0 min-regional movement_00001339_grayscale.h5", (0,650), verbose=True)
    v_out = views.rescale_video(video, 0.5, False)

    views.show_frames(v_out)

    return


if __name__ == "__main__":
    main()