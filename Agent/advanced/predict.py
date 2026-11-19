"""Reload the selected ensemble with its original output specification."""
import argparse
from pathlib import Path

import joblib
from engine import probabilities, export
from agent import read_json


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    model=joblib.load(a.model)
    export(model,a.input,probabilities(model,a.input),read_json(a.task),a.output)
