"""
Integrated script for converting AVI files to centered grayscale NPY files.
Processes AVI input to temporary grayscale NPY chunks, applies centering in parallel,
and aggregates to final NPY output.
"""

import cv2
import numpy as np
import tempfile
import os
from multiprocessing import Process, Queue, cpu_count
import sys
import core.radial_segmentation as rdl
import utils.utils as utils


def produce_frame_chunks(video_path, queue, chunk_size, num_workers):
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
    for _ in range(num_workers):
        queue.put(None)
    cap.release()


def grayscale_worker(queue, temp_dir, worker_id):
    print(f"[Grayscale Worker {worker_id}] Started")
    while True:
        item = queue.get()
        if item is None:
            break

        chunk_idx, frames = item
        print(f"[Grayscale Worker {worker_id}] Processing chunk {chunk_idx} with {len(frames)} frames")
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        temp_path = os.path.join(temp_dir, f"chunk_{chunk_idx:05d}.npy")
        np.save(temp_path, np.array(gray_frames, dtype=np.uint8))
        print(f"[Grayscale Worker {worker_id}] Wrote temp file: {temp_path}")

    print(f"[Grayscale Worker {worker_id}] Exiting")


def centering_worker(temp_files, temp_dir, centered_dir, worker_id):
    print(f"[Centering Worker {worker_id}] Started, processing {len(temp_files)} chunks")
    for temp_file in temp_files:
        chunk_path = os.path.join(temp_dir, temp_file)
        video_chunk = np.load(chunk_path)
        print(f"[Centering Worker {worker_id}] Loaded chunk {temp_file}, shape {video_chunk.shape}")

        # Apply centering
        # centered_chunk, _ = rdl.centre_video_mp(video_chunk)
        centered_chunk, _ = rdl.centre_video(video_chunk)

        # Save centered chunk
        centered_path = os.path.join(centered_dir, temp_file)
        np.save(centered_path, centered_chunk.astype(np.uint8))
        print(f"[Centering Worker {worker_id}] Saved centered chunk: {centered_path}")

    print(f"[Centering Worker {worker_id}] Finished")


def aggregate_centered_chunks(centered_dir, output_path):
    centered_files = sorted(f for f in os.listdir(centered_dir) if f.endswith(".npy"))
    if not centered_files:
        raise ValueError("No centered chunks found")

    # Load all chunks and concatenate
    chunks = []
    for f in centered_files:
        chunk_path = os.path.join(centered_dir, f)
        chunk = np.load(chunk_path)
        chunks.append(chunk)
        print(f"[Aggregator] Loaded chunk {f}")

    full_video = np.concatenate(chunks, axis=0)
    print(f"[Aggregator] Concatenated into array of shape {full_video.shape}")

    np.save(output_path, full_video)
    print(f"[Aggregator] Saved to {output_path}")


def main(video_path, output_path, chunk_size=100, num_workers=cpu_count()):
    print("[Main] Starting integrated processing pipeline")

    # Get frame shape from first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read video for metadata.")
    frame_shape = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="grayscale_chunks_")
    centered_dir = tempfile.mkdtemp(prefix="centered_chunks_")

    try:
        # Grayscale conversion phase
        queue = Queue(maxsize=10)
        grayscale_workers = [
            Process(target=grayscale_worker, args=(queue, temp_dir, i))
            for i in range(num_workers)
        ]
        for w in grayscale_workers:
            w.start()

        print("[Main] Starting frame reader for grayscale conversion")
        produce_frame_chunks(video_path, queue, chunk_size, num_workers)

        for w in grayscale_workers:
            w.join()

        # Get temp files
        temp_files = sorted(f for f in os.listdir(temp_dir) if f.endswith(".npy"))
        print(f"[Main] Grayscale conversion complete, {len(temp_files)} chunks created")
        
        # Centering phase - parallel over chunks
        print("[Main] Starting centering phase")
        num_workers = min(cpu_count(), len(temp_files))
        chunks_per_worker = (len(temp_files) + num_workers - 1) // num_workers  # ceil division
        centering_workers = []

        for i in range(num_workers):
            start_idx = i * chunks_per_worker
            end_idx = min((i + 1) * chunks_per_worker, len(temp_files))
            worker_files = temp_files[start_idx:end_idx]
            w = Process(target=centering_worker, args=(worker_files, temp_dir, centered_dir, i))
            centering_workers.append(w)
            w.start()

        for w in centering_workers:
            w.join()

        print("[Main] Centering complete")

        # Aggregation phase
        print("[Main] Aggregating centered chunks")
        aggregate_centered_chunks(centered_dir, output_path)

        print("[Main] Done.")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(centered_dir, ignore_errors=True)
        print("[Main] Cleaned up temporary directories")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SyntaxError(f"{sys.argv[0]} expects two args: [file_in] [file_out]"
                          f"\n\tExample usage: {sys.argv[0]} video1339.avi video1339_centered.npy")

    file_in = sys.argv[1]
    file_out = sys.argv[2]

    if not os.path.isfile(file_in):
        raise FileNotFoundError("Input video not found.")
    if not file_out.endswith(".npy"):
        raise SyntaxError("Output file must be a .npy file."
                          f"\n\tExample usage: {sys.argv[0]} video1339.avi video1339_centered.npy")

    # Hard-coded chunk_size, can be parameterized if needed
    main(file_in, file_out, chunk_size=100, num_workers=8)
