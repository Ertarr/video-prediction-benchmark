"""
Download the KTH Actions dataset (6 action classes, 25 subjects).
Total size: ~750 MB.

Source: https://www.csc.kth.se/cvap/actions/
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

ACTIONS = [
    "walking", "jogging", "running",
    "boxing", "handwaving", "handclapping",
]

BASE_URL = "https://www.csc.kth.se/cvap/actions"


def download_action(action: str, raw_dir: Path) -> None:
    """Download and extract one action archive."""
    zip_path = raw_dir / f"{action}.zip"
    out_dir = raw_dir / action
    
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  {action}: already extracted, skipping")
        return
    
    if not zip_path.exists():
        url = f"{BASE_URL}/{action}.zip"
        print(f"  {action}: downloading from {url}...")
        urlretrieve(url, zip_path)
    
    print(f"  {action}: extracting to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main():
    raw_dir = Path("data/kth/raw_videos")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading KTH actions to {raw_dir.resolve()}")
    for action in ACTIONS:
        download_action(action, raw_dir)
    
    print("\nDone. Extracted videos:")
    for action in ACTIONS:
        n = len(list((raw_dir / action).glob("*.avi")))
        print(f"  {action}: {n} videos")


if __name__ == "__main__":
    main()