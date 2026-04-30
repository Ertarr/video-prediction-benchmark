"""
Extract frames from KTH .avi videos.

Each video becomes a directory of .png frames at 128x128 grayscale,
matching the standard preprocessing used in the literature
(Villegas et al. 2017, SimVP, etc.).

Naming convention: <action>/person<XX>_<action>_d<X>_uncomp/<frame>.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


ACTIONS = [
    "walking", "jogging", "running",
    "boxing", "handwaving", "handclapping",
]

TARGET_SIZE = 128  # final frame size


def extract_video(video_path: Path, out_dir: Path) -> int:
    """Extract a single .avi to a directory of .png frames. Returns frame count."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return len(list(out_dir.glob("*.png")))
    
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale, resize to 128x128
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE),
                             interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(str(out_dir / f"{frame_idx:04d}.png"), resized)
        frame_idx += 1
    
    cap.release()
    return frame_idx


def main():
    raw_dir = Path("data/kth/raw_videos")
    frames_dir = Path("data/kth/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    total_videos = 0
    total_frames = 0
    
    for action in ACTIONS:
        action_videos = sorted((raw_dir / action).glob("*.avi"))
        print(f"\n{action}: extracting {len(action_videos)} videos...")
        
        for video_path in action_videos:
            video_name = video_path.stem  # e.g. "person01_boxing_d1_uncomp"
            out_dir = frames_dir / action / video_name
            n_frames = extract_video(video_path, out_dir)
            total_videos += 1
            total_frames += n_frames
        
        print(f"  done")
    
    print(f"\nTotal: {total_videos} videos, {total_frames} frames")
    print(f"Disk usage: {frames_dir}")


if __name__ == "__main__":
    main()