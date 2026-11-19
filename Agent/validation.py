"""Split training rows using the evaluation structure declared by a task."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def times(frame, column):
    series = frame[column]
    if series.eq("").any():
        raise ValueError("Missing time values")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    return pd.to_datetime(series, format="mixed", utc=True, errors="raise")


def split_rows(frame, spec, seed, supplied=None):
    kind = spec.get("split", "stratified")
    if kind not in ("stratified", "group", "time"):
        raise ValueError(f"Unsupported split: {kind}")
    fraction = float(spec.get("valid_fraction", 0.25))
    if not 0 < fraction < 1:
        raise ValueError("valid_fraction must be between zero and one")
    group = spec.get("group_column")
    time = spec.get("time_column")
    if group and kind != "group" or time and kind != "time":
        raise ValueError("Choose the split matching group_column or time_column")
    if kind == "group" and not group or kind == "time" and not time:
        raise ValueError("Declare group_column or time_column for this split")
    if supplied is not None:
        train, valid = frame, supplied
    elif kind == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        a, b = next(splitter.split(frame, groups=frame[group]))
        train, valid = frame.iloc[a], frame.iloc[b]
    elif kind == "time":
        values = times(frame, time)
        cut = min(len(frame)-1, max(1, int(len(frame)*(1-fraction))))
        boundary = values.sort_values().iloc[cut]
        train, valid = frame.loc[values < boundary], frame.loc[values >= boundary]
    else:
        train, valid = train_test_split(frame, test_size=fraction, random_state=seed,
                                         stratify=frame[spec["target"]])
    if train.empty or valid.empty:
        raise ValueError("Split produced an empty partition")
    if kind == "group" and (train[group].eq("").any() or valid[group].eq("").any()
                            or set(train[group]).intersection(valid[group])):
        raise ValueError("Missing or overlapping validation groups")
    if kind == "time":
        ordered = times(pd.concat([train,valid],ignore_index=True),time)
        if ordered.iloc[:len(train)].max() >= ordered.iloc[len(train):].min():
            raise ValueError("Validation must strictly follow training time")
    target = spec["target"]
    if train[target].nunique() < 2 or not set(valid[target]).issubset(set(train[target])):
        raise ValueError("Insufficient training classes or unseen validation label")
    if spec["metric"] == "roc_auc" and set(valid[target]) != set(train[target]):
        raise ValueError("ROC AUC needs all classes in validation")
    return train, valid
