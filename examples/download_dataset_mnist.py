import argparse
import gzip
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


MNIST_GZ = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://yann.lecun.org/exdb/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, dst)


def ensure_mnist_gz(raw_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, name in MNIST_GZ.items():
        dst = raw_dir / name
        if dst.exists():
            paths[key] = dst
            continue

        last_err: Exception | None = None
        for base in MIRRORS:
            url = f"{base}/{name}"
            try:
                print(f"Downloading {name} from {base} ...")
                _download(url, dst)
                last_err = None
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_err = e
                continue

        if last_err is not None:
            raise RuntimeError(f"Failed to download {name} from all mirrors.") from last_err

        paths[key] = dst
    return paths


def read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic: {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic: {magic}")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Download MNIST and save train/test as NumPy arrays.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/mnist/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/mnist"))
    args = p.parse_args()

    paths = ensure_mnist_gz(args.raw_dir)

    x_train = read_idx_images_gz(paths["train_images"]).astype(np.uint8, copy=False)
    y_train = read_idx_labels_gz(paths["train_labels"]).astype(np.uint8, copy=False)
    x_test = read_idx_images_gz(paths["test_images"]).astype(np.uint8, copy=False)
    y_test = read_idx_labels_gz(paths["test_labels"]).astype(np.uint8, copy=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.npz"
    test_path = args.out_dir / "test.npz"
    np.savez_compressed(train_path, x=x_train, y=y_train, n=np.array(x_train.shape[0], dtype=np.int64))
    np.savez_compressed(test_path, x=x_test, y=y_test, n=np.array(x_test.shape[0], dtype=np.int64))
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
