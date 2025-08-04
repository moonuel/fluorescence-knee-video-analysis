import cv2
import h5py
import numpy as np
import os
from multiprocessing import Process, Queue
import tempfile
import pathlib

def frame_reader(avi_path, chunk_size, frame_queue, num_workers):
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {avi_path}")

    chunk = []
    chunk_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if chunk:
                frame_queue.put((chunk_id, chunk))
            break

        chunk.append(frame)
        if len(chunk) == chunk_size:
            frame_queue.put((chunk_id, chunk))
            chunk = []
            chunk_id += 1

    # Send sentinel None to workers to signal completion
    for _ in range(num_workers):
        frame_queue.put(None)

    cap.release()

def worker_process(frame_queue, temp_dir, h, w):
    while True:
        item = frame_queue.get()
        if item is None:
            break

        chunk_id, frames = item
        gray_frames = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)

        temp_path = os.path.join(temp_dir, f"chunk_{chunk_id:05d}.h5")
        with h5py.File(temp_path, "w") as f:
            f.create_dataset(
                "video",
                data=np.stack(gray_frames),
                dtype=np.uint8,
                compression="gzip"
            )

def aggregate_temp_files(temp_dir, output_file, total_frames, h, w, chunk_size):
    chunk_files = sorted(os.listdir(temp_dir))
    with h5py.File(output_file, "w") as h5f:
        dset = h5f.create_dataset(
            "video",
            shape=(total_frames, h, w),
            dtype=np.uint8,
            chunks=(chunk_size, h, w),
            compression="gzip"
        )

        start_idx = 0
        for chunk_file in chunk_files:
            path = os.path.join(temp_dir, chunk_file)
            with h5py.File(path, "r") as temp_f:
                data = temp_f["video"][:]
                n = data.shape[0]
                dset[start_idx : start_idx + n] = data
                start_idx += n
            os.remove(path)

def convert_avi_to_hdf5_grayscale(
    avi_path: str,
    output_dir: str,
    chunk_size: int = 200,
    num_workers: int = 4
    ) -> str:
    avi_path = pathlib.Path(avi_path).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (avi_path.stem + "_grayscale.h5")

    # Get video metadata
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {avi_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    frame_queue = Queue(maxsize=num_workers * 2)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Start worker processes
        workers = []
        for _ in range(num_workers):
            p = Process(target=worker_process, args=(frame_queue, temp_dir, h, w))
            p.start()
            workers.append(p)

        # Run reader in main process
        frame_reader(str(avi_path), chunk_size, frame_queue, num_workers)

        # Wait for all workers to finish
        for p in workers:
            p.join()

        # Aggregate chunks into final file
        aggregate_temp_files(temp_dir, str(output_file), total_frames, h, w, chunk_size)

    print(f"Grayscale video saved to: {output_file}")
    return str(output_file)


if __name__ == "__main__":

    avi_path = "../data/raw/1 con-0 min-fluid movement_00001190.avi"
    output_dir = "../data/raw/tmp"
    convert_avi_to_hdf5_grayscale(avi_path, output_dir, chunk_size=200, num_workers=8)
