"""
Pack the extracted KTH frames into one uint8 array on disk.

Reading and decoding ~210,000 PNG files per epoch made the KTH runs
CPU-bound (about 5 minutes per epoch with 5 runs in parallel on Colab).
This script decodes every frame once and writes:
    data/kth/frames_uint8.npy     (n_frames, 128, 128) uint8
    data/kth/frames_index.json    {"walking/person01_walking_d1_uncomp": [offset, n_frames], ...}
KTHActions uses them automatically when they exist (same pixel values,
read straight from a memory-mapped file instead of PNG).

Usage:
    python scripts/kth_to_npy.py --kth-root data/kth
"""
import sys
import json
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image


def load_video(video_dir: Path) -> np.ndarray:
    frames = sorted(video_dir.glob("*.png"))
    return np.stack([np.array(Image.open(f), dtype=np.uint8) for f in frames])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kth-root", default="data/kth")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frames_root = Path(args.kth_root) / "frames"
    videos = sorted(p for p in frames_root.glob("*/*") if p.is_dir())
    counts = [len(list(v.glob("*.png"))) for v in videos]
    total = sum(counts)
    print(f"{len(videos)} videos, {total} frames")

    out = np.lib.format.open_memmap(Path(args.kth_root) / "frames_uint8.npy", mode="w+",
                                    dtype=np.uint8, shape=(total, 128, 128))
    index, offset = {}, 0
    with Pool(args.workers) as pool:
        for video, n, arr in zip(videos, counts, pool.imap(load_video, videos, chunksize=4)):
            assert arr.shape == (n, 128, 128), (video, arr.shape)
            out[offset:offset + n] = arr
            index[video.relative_to(frames_root).as_posix()] = [offset, n]
            offset += n
    out.flush()
    (Path(args.kth_root) / "frames_index.json").write_text(json.dumps(index))
    print(f"wrote {offset} frames to {Path(args.kth_root) / 'frames_uint8.npy'}")


if __name__ == "__main__":
    sys.exit(main())
