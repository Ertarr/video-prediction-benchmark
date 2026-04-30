"""
Generate train/val/test split files for KTH.

Standard protocol:
    train:  persons 01-13
    val:    persons 14-16
    test:   persons 17-25 (the standard literature test split)

Each split is a text file listing clips, one per line:
    <action>/<video_name> <start_frame> <end_frame>
"""
from pathlib import Path
from collections import defaultdict


ACTIONS = [
    "walking", "jogging", "running",
    "boxing", "handwaving", "handclapping",
]

TRAIN_PERSONS = list(range(1, 14))   # 01-13
VAL_PERSONS = list(range(14, 17))    # 14-16
TEST_PERSONS = list(range(17, 26))   # 17-25

CLIP_LEN = 30           # 10 input + 20 target frames per clip
STRIDE = 20             # overlap or not between clips for more training data
MIN_VIDEO_FRAMES = 30   # discard videos shorter than one clip


def parse_person_id(video_name: str) -> int:
    """Extract person ID from a video name like 'person01_boxing_d1_uncomp'."""
    # video_name starts with "personXX..."
    return int(video_name[6:8])


def get_clips_for_video(video_dir: Path) -> list[tuple[int, int]]:
    """Get a list of (start, end) frame indices for clips in a video."""
    n_frames = len(list(video_dir.glob("*.png")))
    if n_frames < MIN_VIDEO_FRAMES:
        return []
    
    clips = []
    start = 0
    while start + CLIP_LEN <= n_frames:
        clips.append((start, start + CLIP_LEN))
        start += STRIDE
    return clips


def assign_split(person_id: int) -> str | None:
    """Return 'train', 'val', 'test', or None."""
    if person_id in TRAIN_PERSONS:
        return "train"
    if person_id in VAL_PERSONS:
        return "val"
    if person_id in TEST_PERSONS:
        return "test"
    return None


def main():
    frames_dir = Path("data/kth/frames")
    splits_dir = Path("data/kth")
    
    split_clips = defaultdict(list)
    
    for action in ACTIONS:
        action_dir = frames_dir / action
        if not action_dir.exists():
            print(f"  warning: {action_dir} not found, skipping")
            continue
        
        for video_dir in sorted(action_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            
            person_id = parse_person_id(video_dir.name)
            split = assign_split(person_id)
            if split is None:
                continue
            
            clips = get_clips_for_video(video_dir)
            for (start, end) in clips:
                rel_path = f"{action}/{video_dir.name}"
                split_clips[split].append(f"{rel_path} {start} {end}")
    
    for split, lines in split_clips.items():
        out_path = splits_dir / f"{split}_clips.txt"
        out_path.write_text("\n".join(lines))
        print(f"  {split}: {len(lines)} clips written to {out_path}")


if __name__ == "__main__":
    main()