from src.utils import utils, io, views
from src.core import knee_segmentation as ks
from src.core import data_processing as dp
from src.config import VERBOSE

def main():
    if VERBOSE: print("main() called!")

    # Load example video
    video = io.load_nparray("../data/processed/aging_knee_processed.npy")

    # Show video
    views.show_frames(video)

if __name__ == "__main__":
    main()