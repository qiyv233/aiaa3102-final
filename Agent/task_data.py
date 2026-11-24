"""Read classification splits from a directory or a course data archive."""

from contextlib import contextmanager
import csv
import io
import json
import math
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


def image_inputs(frame, directory):
    """Keep on-disk paths or load ZIP members into seekable streams for Pillow."""
    if len(frame.columns) != 1:
        raise ValueError("Image manifests require one path column")
    column = frame.columns[0]

    def resource(value):
        if isinstance(value, io.BytesIO):
            return value
        path = data_file(directory, value)
        if isinstance(path, Path):
            return str(path.resolve())
        return io.BytesIO(path.read_bytes())

    frame[column] = frame[column].map(resource)
    return frame


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


def jsonl_frame(stream, required=(), expected_types=None):
    """Preserve string identifiers and distinguish Boolean values from string labels."""
    def unique_fields(pairs):
        record = dict(pairs)
        if len(record) != len(pairs):
            raise ValueError("Duplicate JSON object key")
        return record

    records = []
    kinds = {}
    for number, raw in enumerate(stream, 1):
        try:
            line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique_fields)
            if not isinstance(row, dict) or not row or any(not k.strip() for k in row):
                raise ValueError("Expected a flat object with named fields")
            if any(row.get(k) is None for k in required):
                raise ValueError("Required field absent or null")
            for key, value in row.items():
                if value is None:
                    continue
                if type(value) is bool:
                    kind = "boolean"
                elif isinstance(value, str):
                    kind = "string"
                elif isinstance(value, (int,float)) and (type(value) is int or math.isfinite(value)):
                    kind = "number"
                else:
                    raise ValueError(f"Unsupported value in {key!r}")
                expected = (expected_types or {}).get(key)
                if kinds.setdefault(key,kind) != kind or (expected is not None and kind != expected):
                    raise ValueError(f"Inconsistent JSON type for {key!r}")
                if kind != "string":
                    row[key] = json.dumps(value)
            records.append(row)
        except ValueError as exc:
            raise ValueError(f"JSONL line {number}: {exc}") from exc
    if not records:
        raise ValueError("Empty JSONL input")
    return pd.DataFrame.from_records(records), kinds


def headerless_frame(stream, spec, *, labeled):
    columns = spec["tsv_columns"]
    inputs = spec["features"]
    if len(set(columns)) != len(columns) or set(columns) != {spec["target"], *inputs}:
        raise ValueError("TSV column order disagrees with the declared inputs and target")
    with io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
        rows = [row for row in csv.reader(text, delimiter="\t", strict=True) if row]
    if not rows:
        raise ValueError("Empty headerless TSV")
    names = inputs if not labeled and len(rows[0]) == len(inputs) else columns
    if any(len(row) != len(names) for row in rows):
        raise ValueError("TSV row width disagrees with the publisher's schema")
    return pd.DataFrame(rows, columns=names)


def load_features(path, spec):
    if path.suffix.lower() in {".jsonl",".ndjson"}:
        with path.open("rb") as stream:
            return jsonl_frame(stream,spec["features"],spec.get("json_types"))[0]
    if path.suffix.lower() == ".tsv" and spec.get("native_format") == "headerless_tsv":
        with path.open("rb") as stream:
            return headerless_frame(stream, spec, labeled=False)
    if path.suffix.lower()==".parquet":
        from native_formats import read_parquet
        return read_parquet(path).fillna("").astype(str)
    if path.suffix.lower()==".gz" and spec.get("native_format")=="idx":
        from native_formats import image_frame
        return image_frame(path,spec["features"][0])
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
    if path.suffix.lower()==".gz":
        from native_formats import idx
        labels=idx(path,1)
        if len(labels)!=rows:
            raise ValueError("Number of IDX labels does not match images")
        return pd.Series(labels).astype(str)
    with path.open("rb") as stream:
        labels = np.load(stream, allow_pickle=False)
    if labels.shape not in ((rows,), (rows, 1)):
        raise ValueError(f"Number of labels does not match the feature rows: {path}")
    return pd.Series(labels.reshape(-1)).fillna("").astype(str)


def load_labeled(paths, spec):
    path, label_path = paths
    if path.suffix.lower() in {".jsonl",".ndjson"}:
        if label_path is not None:
            raise ValueError("JSONL classification expects labels in the records")
        with path.open("rb") as stream:
            return jsonl_frame(stream,[*spec["features"],spec["target"]],spec.get("json_types"))[0]
    if path.suffix.lower() == ".tsv" and spec.get("native_format") == "headerless_tsv":
        if label_path is not None:
            raise ValueError("The publisher's TSV already declares a label column")
        with path.open("rb") as stream:
            return headerless_frame(stream, spec, labeled=True)
    frame = load_features(path, spec)
    if label_path is not None:
        frame[spec["target"]] = load_labels(label_path, len(frame))
    return frame
