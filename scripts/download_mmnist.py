import urllib.request
from pathlib import Path
import os

if Path.cwd().name == "scripts":
    os.chdir("..")

URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
DEST = Path("data/moving_mnist/mnist_test_seq.npy")

def download():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already downloaded at {DEST}")
        return
    print(f"Downloading from {URL}...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST}")

if __name__ == "__main__":
    download()