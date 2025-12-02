"""Reads, centers, and saves the raw video file."""

import utils.io as io

def main():

    # fn = 
    io.convert_avi_to_hdf5_grayscale(f"../data/raw/1 con-0 min-fluid movement_00001190.avi", "../data/raw/")

    return


if __name__ == "__main__":
    main()