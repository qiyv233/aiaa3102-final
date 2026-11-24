"""Exercise image training and interchangeable ZIP/directory replay on public digits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    dataset = out / "digits"
    images = dataset / "images"
    images.mkdir(parents=True)
    records, labels = load_digits(return_X_y=True)
    train, val = train_test_split(np.arange(len(labels)), test_size=.2,
                                  stratify=labels, random_state=20261124)
    for i, pixels in enumerate(records):
        image = np.rint(pixels.reshape(8, 8) * 255 / 16).astype("uint8")
        Image.fromarray(image).save(images / f"img{i:04d}.png")
    for split, rows in (("train", train), ("val", val)):
        pd.DataFrame({"image": [f"images/img{i:04d}.png" for i in rows],
                      "label": labels[rows]}).to_csv(dataset / (split + ".csv"), index=False)
    (dataset / "README.md").write_text(
        "Classify each grayscale image into one of ten classes.\n\n"
        "Training data: train.csv. Validation data: val.csv.\n\n"
        "Input feature: image. Target column: label.\n\nEvaluation metric: accuracy.\n", encoding="utf-8")
    archive = out / "digits.zip"
    with ZipFile(archive, "w") as bundle:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(dataset).as_posix())

    cli = [sys.executable, "-B", str(ROOT/"Agent/agent.py")]
    reference = None
    reference_partition = None
    for name, source, other in (("directory", dataset, archive), ("archive", archive, dataset)):
        folder = out / name
        subprocess.run([*cli, "develop", "--data", str(source), "--out", str(folder),
                        "--strengthen", "--max-actions", "3", "--provider", "local",
                        "--train-only", "--seconds", "600"], check=True)
        subprocess.run([*cli, "validate", "--data", str(source), "--out", str(folder)], check=True)
        result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        assert result["stage"] == "full_refit" and result["run_status"] == "completed"
        assert result["train_rows"] == len(train) and result["validation_rows"] == len(val)
        assert result["charged_calls"] == 0
        replay = folder / "cross-format.csv"
        subprocess.run([*cli, "predict", "--model", str(folder/"model.joblib"), "--data", str(other),
                        "--input", "val.csv", "--output", str(replay)], check=True)
        prediction = (folder/"validation-predictions.csv").read_bytes()
        partition = (folder/"partition.csv").read_bytes()
        assert replay.read_bytes() == prediction
        if reference is not None:
            assert reference == prediction and reference_partition == partition
        reference, reference_partition = prediction, partition
    print(f"Identical directory/ZIP partitions, predictions and both cross-format replays on {len(val)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    run(parser.parse_args().out)
