"""Read classification splits from a directory or a course data archive."""

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import Path as ZipPath, ZipFile

import numpy as np
import pandas as pd


@contextmanager
def data_root(source):
    if source.suffix.lower() == ".zip" and not source.is_dir():
        with ZipFile(source) as archive:
            yield ZipPath(archive)
    else:
        yield source


def data_file(root, name):
    if isinstance(root, Path):
        return root / name
    relative = PurePosixPath(Path(name).as_posix())
    if relative.is_absolute() or Path(name).drive or ".." in relative.parts:
        raise ValueError(f"Archive entry must be relative to its root: {name}")
    return root.joinpath(*relative.parts)


def task_files(root, spec):
    """Resolve split paths and reject reuse of an input or label file across splits."""
    files = {}
    seen = set()
    for split in ("train", "validation", "test"):
        if not spec.get(split):
            continue
        paths = tuple(data_file(root, spec[key]) if key in spec else None
                      for key in (split, f"{split}_labels"))
        for path in paths:
            if path is None:
                continue
            identity = path.resolve() if isinstance(path, Path) else path.at
            if identity in seen:
                raise ValueError(f"Split files must be distinct: {path}")
            seen.add(identity)
        files[split] = paths
    return files


def load_csv(path):
    # Keep leading zeros in IDs and preserve label names.
    if isinstance(path, ZipPath):
        with path.open("rb") as stream:
            return pd.read_csv(stream, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_features(path, spec):
    if path.suffix.lower() != ".npy":
        return load_csv(path)
    columns = spec.get("features")
    if not columns:
        raise ValueError("NumPy feature arrays require named features in column order")
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=spec.get("allow_pickle_features") is True)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"NumPy array shape does not match declared features: {path}")
    return pd.DataFrame(values, columns=columns).fillna("").astype(str)


def load_labels(path, rows):
    with path.open("rb") as stream:
        labels = np.load(stream, allow_pickle=False)
    if labels.shape not in ((rows,), (rows, 1)):
        raise ValueError(f"Number of labels does not match the feature rows: {path}")
    return pd.Series(labels.reshape(-1)).fillna("").astype(str)


def load_labeled(paths, spec):
    path, label_path = paths
    frame = load_features(path, spec)
    if label_path is not None:
        frame[spec["target"]] = load_labels(label_path, len(frame))
    return frame
