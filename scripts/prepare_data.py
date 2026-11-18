"""Decode Stanford Cars parquet files into uint8 arrays and make a fixed train/validation split.

Input : data/raw/{train,test}-*.parquet  (Hugging Face mirror tanganke/stanford_cars; --download fetches them)
Output: data/cars_{train,test}_x.npy  uint8 [N, S, S, 3]
        data/cars_{train,test}_y.npy  int64 [N]
        data/cars_split.npz           train_idx, val_idx (indices into the train arrays)
"""
import argparse
import glob
import io
import os
import urllib.request
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SIZE = 256
MIRROR = "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data"
FILES = ["train-00000-of-00002", "train-00001-of-00002", "test-00000-of-00002", "test-00001-of-00002"]


def download(raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for name in FILES:
        path = os.path.join(raw_dir, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"downloading {name}.parquet")
            urllib.request.urlretrieve(f"{MIRROR}/{name}.parquet", path)


def _decode(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def convert(split: str, raw_dir: str, out_dir: str, workers: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(raw_dir, f"{split}-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files for split '{split}' in {raw_dir}")
    raws, labels = [], []
    for f in files:
        table = pq.read_table(f, columns=["image", "label"]).to_pydict()
        raws += [im["bytes"] for im in table["image"]]
        labels += table["label"]
    with Pool(workers) as pool:
        x = np.stack(pool.map(_decode, raws, chunksize=64))
    y = np.asarray(labels, dtype=np.int64)
    np.save(os.path.join(out_dir, f"cars_{split}_x.npy"), x)
    np.save(os.path.join(out_dir, f"cars_{split}_y.npy"), y)
    print(f"{split}: {x.shape} images, {len(np.unique(y))} classes")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--download", action="store_true", help="fetch the parquet files first (about 1.9 GB)")
    a = ap.parse_args()

    if a.download:
        download(a.raw)

    y = convert("train", a.raw, a.out, a.workers)
    convert("test", a.raw, a.out, a.workers)

    # Stratified split: the same fraction of every class goes to validation.
    rng = np.random.default_rng(a.split_seed)
    val = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        val += idx[: max(1, round(len(idx) * a.val_frac))].tolist()
    val_idx = np.sort(np.asarray(val))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    np.savez(os.path.join(a.out, "cars_split.npz"), train_idx=train_idx, val_idx=val_idx)
    print(f"split: {len(train_idx)} train, {len(val_idx)} validation")


if __name__ == "__main__":
    main()
