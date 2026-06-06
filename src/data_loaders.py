"""
Dataset loaders for all 9 benchmark datasets.

Active (3 light): glass+identification, ecoli, Breast_Cancer
Stub/disabled  : coil-20-proc, Digits, image+segmentation, coil-100, MNIST, covertype

Each loader returns (X: np.ndarray, y: np.ndarray) with:
  X : float64 (n_samples, n_features)  — raw, BEFORE normalization
  y : int     (n_samples,)             — 0..k-1
"""

from __future__ import annotations

import gzip
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing import encode_labels

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN columns and fill remaining NaN with column median."""
    df = df.dropna(axis=1, how="all")
    return df.fillna(df.median(numeric_only=True))


def _load_indexed(path: Path, sep: str) -> pd.DataFrame:
    """Load a file without header; sep = ',' or whitespace regex."""
    return pd.read_csv(path, sep=sep, header=None, engine="python")


def _indexed_to_Xy(df: pd.DataFrame, label_col: int, drop_cols: list[int]) -> tuple:
    n_cols = df.shape[1]
    if label_col < 0:
        label_col = n_cols + label_col
    y = encode_labels(df.iloc[:, label_col].values)
    skip = set([label_col] + [d if d >= 0 else n_cols + d for d in drop_cols])
    feat = [c for c in range(n_cols) if c not in skip]
    X = df.iloc[:, feat].values.astype(float)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ① glass+identification
# ──────────────────────────────────────────────────────────────────────────────

def load_glass(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / cfg["path"]
    logger.debug(f"Loading glass from {path}")
    df = _load_indexed(path, sep=",")
    X, y = _indexed_to_Xy(df, label_col=-1, drop_cols=[0])
    logger.info(f"glass+identification: n={X.shape[0]}, d={X.shape[1]}, classes={np.unique(y).tolist()}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ② ecoli
# ──────────────────────────────────────────────────────────────────────────────

def load_ecoli(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / cfg["path"]
    logger.debug(f"Loading ecoli from {path}")
    df = _load_indexed(path, sep=r"\s+")
    X, y = _indexed_to_Xy(df, label_col=-1, drop_cols=[0])
    logger.info(f"ecoli: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ③ Breast_Cancer
# ──────────────────────────────────────────────────────────────────────────────

def load_breast_cancer(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / cfg["path"]
    logger.debug(f"Loading Breast_Cancer from {path}")
    df = pd.read_csv(path)
    label_col = cfg["label_col"]      # "diagnosis"
    drop_cols  = cfg.get("drop_cols", [])
    y = encode_labels(df[label_col].values)
    df = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns])
    df = _clean_csv(df)
    X = df.values.astype(float)
    logger.info(f"Breast_Cancer: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ④ image+segmentation (7 classes, n=2310 from test file)
# ──────────────────────────────────────────────────────────────────────────────

def load_segmentation(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / cfg["path"]
    skip_rows = cfg.get("skip_rows", 5)
    logger.debug(f"Loading segmentation from {path}")
    df = pd.read_csv(path, header=None, skiprows=skip_rows)
    # First column = class name (string), rest = features
    y = encode_labels(df.iloc[:, 0].values)
    X = df.iloc[:, 1:].values.astype(float)
    logger.info(f"image+segmentation: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ coil-20-proc / coil-100 (image folders)
# ──────────────────────────────────────────────────────────────────────────────

def load_coil(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    folder = data_dir / cfg["path"]
    image_size = tuple(cfg.get("image_size", [32, 32]))
    image_mode = cfg.get("image_mode", "L")
    logger.debug(f"Loading COIL from {folder}, size={image_size}, mode={image_mode}")

    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")

    Xs, ys = [], []
    for f in files:
        # Filename: obj<class>__<view>.png
        stem = f.stem  # e.g. "obj10__0"
        try:
            class_id = int(stem.split("__")[0].replace("obj", "")) - 1
        except (ValueError, IndexError):
            continue
        img = Image.open(f).convert(image_mode).resize(image_size[::-1])
        Xs.append(np.asarray(img, dtype=float).ravel())
        ys.append(class_id)

    X = np.stack(Xs)
    y = np.array(ys, dtype=int)
    logger.info(f"COIL: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ Digits (sklearn built-in, n=1797, d=64, c=10)
# ──────────────────────────────────────────────────────────────────────────────

def load_sklearn_digits(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    logger.debug("Loading sklearn Digits dataset")
    bunch = load_digits()
    X = bunch.data.astype(float)
    y = bunch.target.astype(int)
    logger.info(f"Digits: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ MNIST (IDX binary format, train + test = 70000)
# ──────────────────────────────────────────────────────────────────────────────

def _read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 0x00000803:
            raise ValueError(f"Not an MNIST image file: magic={magic}")
        n, h, w = struct.unpack(">III", f.read(12))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h * w)
    return data.astype(float)


def _read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 0x00000801:
            raise ValueError(f"Not an MNIST label file: magic={magic}")
        n = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(map(str, paths)))


def load_mnist(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    folder = data_dir / cfg["path"]
    logger.debug(f"Loading MNIST from {folder}")

    train_imgs_path = _first_existing([
        folder / "train-images.idx3-ubyte",
        folder / "train-images-idx3-ubyte" / "train-images.idx3-ubyte",
        folder / "train-images-idx3-ubyte" / "train-images-idx3-ubyte",
    ])
    train_lbls_path = _first_existing([
        folder / "train-labels.idx1-ubyte",
        folder / "train-labels-idx1-ubyte" / "train-labels.idx1-ubyte",
        folder / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte",
    ])
    test_imgs_path = _first_existing([
        folder / "t10k-images.idx3-ubyte",
        folder / "t10k-images-idx3-ubyte" / "t10k-images.idx3-ubyte",
        folder / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte",
    ])
    test_lbls_path = _first_existing([
        folder / "t10k-labels.idx1-ubyte",
        folder / "t10k-labels-idx1-ubyte" / "t10k-labels.idx1-ubyte",
        folder / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte",
    ])

    train_imgs = _read_idx_images(train_imgs_path)
    train_lbls = _read_idx_labels(train_lbls_path)
    test_imgs  = _read_idx_images(test_imgs_path)
    test_lbls  = _read_idx_labels(test_lbls_path)

    X = np.concatenate([train_imgs, test_imgs], axis=0)
    y = np.concatenate([train_lbls, test_lbls], axis=0)
    logger.info(f"MNIST: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# ⑧ covertype (581012 samples, 54 features, 7 classes)
# ──────────────────────────────────────────────────────────────────────────────

def load_covertype(data_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / cfg["path"]
    logger.debug(f"Loading covertype from {path}")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    y = encode_labels(df.iloc[:, -1].values)
    X = df.iloc[:, :-1].values.astype(float)
    logger.info(f"covertype: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Public dispatch
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir_str: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from config entry.

    Returns
    -------
    X : float64 (n, d) — raw features, BEFORE normalization
    y : int     (n,)   — labels 0..k-1
    """
    data_dir = Path(data_dir_str)
    fmt      = cfg.get("format", "csv")
    name     = cfg.get("name", "?")

    dispatch = {
        "csv":            load_breast_cancer,
        "whitespace":     load_ecoli,
        "csv_noheader":   load_glass,
        "segmentation":   load_segmentation,
        "image":          load_coil,
        "sklearn_digits": load_sklearn_digits,
        "mnist":          load_mnist,
        "covertype":      load_covertype,
    }

    # More general dispatch: route by name for uniquely-named formats
    if fmt not in dispatch:
        raise ValueError(f"Unknown format '{fmt}' for dataset '{name}'")

    loader = dispatch[fmt]
    X, y = loader(data_dir, cfg)
    return X, y
