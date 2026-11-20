"""Export task predictions in the order and schema of an optional sample submission."""
import os
from pathlib import Path

import numpy as np

from task_data import load_csv


def write_submission(frame, dest, template=None, id_column=None):
    if len(set(frame.columns)) != len(frame.columns) or frame.isna().any().any():
        raise ValueError("Duplicate output columns or missing predictions")
    if template is not None:
        sample = load_csv(template)
        if len(sample) != len(frame) or set(sample.columns) != set(frame.columns):
            raise ValueError("Output schema or row count differs from submission template")
        if id_column:
            expected, actual = sample[id_column], frame[id_column]
            if (expected.duplicated().any() or actual.duplicated().any() or expected.eq("").any()
                    or set(expected) != set(actual)):
                raise ValueError("Duplicate, missing or unmatched submission IDs")
            frame = frame.set_index(id_column).loc[expected].reset_index()
        frame = frame[list(sample.columns)]
    tmp = dest.with_suffix(dest.suffix+".tmp")
    frame.to_csv(tmp,index=False,encoding="utf-8")
    os.replace(tmp,dest)


def validate_probabilities(values, classes):
    probs=np.asarray(values,dtype=np.float64)
    total=probs.sum(axis=1,keepdims=True)
    if (probs.shape[1] != len(classes) or not np.isfinite(probs).all()
            or (probs<0).any() or not np.allclose(total,1,atol=1e-5)):
        raise ValueError("Invalid predicted class probabilities")
    return probs/total
