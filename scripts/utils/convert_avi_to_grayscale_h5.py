"""
Reusable utility script for quickly converting large .avi files to .h5 files for chunked data retrieval. 

Parameters are hard-coded and must be set manually.

"""

import cv2
import numpy as np
import h5py
import tempfile
import os
from multiprocessing import Process, Queue, cpu_count
import sys


def frame_reader(video_path, queue, chunk_size):
    cap = cv2.VideoCapture(video_path)
    chunk_idx = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        if len(frames) == chunk_size:
            print(f"[Reader] Read chunk {chunk_idx} with {len(frames)} frames")
            queue.put((chunk_idx, frames))
            chunk_idx += 1
            frames = []

    if frames:
        print(f"[Reader] Read final chunk {chunk_idx} with {len(frames)} frames")
        queue.put((chunk_idx, frames))

    print("[Reader] Finished reading video and pushed final chunk")
    for _ in range(cpu_count()):
        queue.put(None)
    cap.release()


def worker_process(queue, temp_dir, worker_id):
    print(f"[Worker {worker_id}] Started")
    while True:
        item = queue.get()
        if item is None:
            break

        chunk_idx, frames = item
        print(f"[Worker {worker_id}] Processing chunk {chunk_idx} with {len(frames)} frames")
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        temp_path = os.path.join(temp_dir, f"chunk_{chunk_idx:05d}.npy")
        np.save(temp_path, np.array(gray_frames, dtype=np.uint8))
        print(f"[Worker {worker_id}] Wrote temp file: {temp_path}")

    print(f"[Worker {worker_id}] Exiting")


def aggregate_temp_files(temp_dir, output_path, frame_shape):
    temp_files = sorted(f for f in os.listdir(temp_dir) if f.endswith(".npy"))
    total_frames = sum(np.load(os.path.join(temp_dir, f)).shape[0] for f in temp_files)

    print(f"[Aggregator] Aggregating {len(temp_files)} temp files into {output_path}")
    with h5py.File(output_path, "w") as h5f:
        dset = h5f.create_dataset("video", shape=(total_frames, *frame_shape), dtype=np.uint8)
        idx = 0
        for chunk_idx, f in enumerate(temp_files):
            temp_path = os.path.join(temp_dir, f)
            frames = np.load(temp_path)
            print(f"[Aggregator] Writing chunk {chunk_idx} from {f}")
            dset[idx:idx+frames.shape[0]] = frames
            idx += frames.shape[0]
    print(f"[Aggregator] Done writing to {output_path}")


def main(video_path, output_path, chunk_size=100):
    print("[Main] Starting processing pipeline")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read video for metadata.")
    frame_shape = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape

    # queue = Queue(maxsize=cpu_count())
    queue = Queue(maxsize=10)
    temp_dir = tempfile.mkdtemp()

    print("[Main] Spawning worker processes")
    workers = [
        Process(target=worker_process, args=(queue, temp_dir, i))
        for i in range(cpu_count())
    ]
    for w in workers:
        w.start()

    print("[Main] Starting frame reader")
    frame_reader(video_path, queue, chunk_size)

    for w in workers:
        w.join()

    print("[Main] Aggregating temp files")
    aggregate_temp_files(temp_dir, output_path, frame_shape)

    print("[Main] Done.")


# Example usage:
# main("input_video.mp4", "output_video.h5")

if __name__ == "__main__":

    if len(sys.argv) != 3: raise SyntaxError(f"{sys.argv[0]} expects two args: [file_in] [file_out]"
                                             "\n\tExample usage: {sys.argv[0]} video1339.avi video1339gray.h5")

    file_in = sys.argv[1]
    file_out = sys.argv[2]

    if not os.path.isfile(file_in): raise FileNotFoundError("Input video not found.")
    if not file_out[-3:] == ".h5": raise SyntaxError(f"Output file isn't a .h5 file. "
                                                     "\n\tExample usage: {sys.argv[0]} video1339.avi video1339gray.h5")

    # avi_path = "../data/raw/dmm-0 min-fluid movement_00001207.avi"
    # h5_out_path = avi_path[:-4] + ".h5"

    breakpoint()

    main(file_in, file_out, chunk_size=100)

