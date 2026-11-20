"""Offline regression checks. Uses disposable task data and an isolated budget database."""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import numpy as np

import agent
import budget
import planning
from paired_text import PAIR_CANDIDATES, pair_estimator
from validation import split_rows


def verify_pairwise(scratch, cli):
    folder = scratch / "pairwise"
    folder.mkdir()
    pairs = [(f"Welcome to the bright park {i}!", f"Park notes from day {i}") for i in range(8)]
    swapped = [(b, a) for a, b in pairs]
    features = ["left", "right"]
    frame = pd.DataFrame(pairs + swapped, columns=features)
    labels = pd.Series(["1"] * len(pairs) + ["0"] * len(swapped))
    reverse = frame[features[::-1]].set_axis(features, axis=1)
    for name in PAIR_CANDIDATES:
        model = pair_estimator(name, seed=17).fit(frame, labels)
        np.testing.assert_allclose(model.predict_proba(frame), model.predict_proba(reverse)[:, ::-1])

    frame.assign(label=labels).to_csv(folder / "train.csv", index=False)
    spec = dict(train="train.csv", test="withheld.csv", features=features, target="label",
                pairwise=True, metric="accuracy")
    task = folder / "task.json"
    task.write_text(json.dumps(spec), encoding="utf-8")
    output = folder / "out"
    subprocess.run(cli + ["run", "--task", str(task), "--out", str(output),
                         "--strategy", "baseline", "--validation-only"], check=True)
    assert (output / "model.joblib").is_file()
    assert not (output / "predictions.csv").exists()


def main():
    root = Path(__file__).resolve().parent
    with TemporaryDirectory(dir=Path.cwd()) as temp:
        scratch = Path(temp).resolve()
        os.environ["AIAA_BUDGET_DB"] = str(scratch/"budget.sqlite")
        os.environ["TEST_PLANNER_TOKEN"] = "demo"
        frame = pd.DataFrame({"id":range(40),"cluster":[str(i//4) for i in range(40)],
                              "date":[str(i//2) for i in range(40)],"value":[i%5 for i in range(40)],
                              "label":[str(i%2) for i in range(40)]})
        cli=[sys.executable,str(root/"agent.py")]
        verify_pairwise(scratch, cli)
        for kind,col in (("group","cluster"),("time","date")):
            spec=dict(train="train.csv",test="test.csv",id="id",target="label",split=kind,
                      metric="macro_f1",features=["value"])
            spec["group_column" if kind=="group" else "time_column"]=col
            train,valid=split_rows(frame,spec,17)
            if kind=="group":
                assert set(train[col]).isdisjoint(valid[col])
            else:
                assert train[col].astype(int).max()<valid[col].astype(int).min()
            try:
                split_rows(train,spec,17,train)
            except ValueError:
                pass
            else:
                raise AssertionError("Leaking supplied validation accepted")
            folder=scratch/kind
            folder.mkdir()
            if kind == "group":
                spec["submission_template"]="sample.csv"
                pd.DataFrame({"label":[""]*len(frame),"id":frame.id[::-1]}).to_csv(folder/"sample.csv",index=False)
            frame.to_csv(folder/"train.csv",index=False)
            frame.drop(columns="label").to_csv(folder/"test.csv",index=False)
            (folder/"task.json").write_text(json.dumps(spec))
            subprocess.run(cli+["run","--task",str(folder/"task.json"),"--out",str(folder/"out"),
                               "--strategy","baseline"],check=True)
            pred=pd.read_csv(folder/"out/predictions.csv")
            expected=frame.id[::-1] if kind=="group" else frame.id
            assert pred.id.tolist()==expected.tolist()
            if kind=="group":
                assert pred.columns.tolist()==["label","id"]
                subprocess.run(cli+["predict","--model",str(folder/"out/model.joblib"),
                                    "--input",str(folder/"test.csv"),"--template",str(folder/"sample.csv"),
                                    "--output",str(folder/"replay.csv")],check=True)
                assert (folder/"replay.csv").read_bytes()==(folder/"out/predictions.csv").read_bytes()
        # Simulate a stalled fit after a real checkpoint has already been published.
        original_run=subprocess.run
        dest=scratch/"timed"
        def stalled(command, **options):
            shutil.copy(scratch/"time/out/model.joblib",dest/"model.joblib")
            return original_run([sys.executable,"-c","import time; time.sleep(30)"],**options)
        with patch.object(agent.subprocess,"run",side_effect=stalled):
            agent.supervise(argparse.Namespace(seconds=0.3,out=dest,task=scratch/"time/task.json"))
        assert json.loads((dest/"run.jsonl").read_text().splitlines()[-1])["status"]=="training_timeout"
        assert (dest/"predictions.csv").read_bytes()==(scratch/"time/out/predictions.csv").read_bytes()
        cfg=dict(endpoint="https://example.invalid/chat/completions",model="deepseek-test",
                 key_env="TEST_PLANNER_TOKEN",release_date="2026-09-01",price_date="2026-10-21",
                 price_source="simulated",input_rmb_per_million=10,output_rmb_per_million=10,
                 official_context_limit=1000000,max_output=50)
        reply={"id":"sim","usage":{"prompt_tokens":100,"completion_tokens":20},
               "choices":[{"message":{"content":json.dumps({"action":"a","reason":"fixture"})}}]}
        with patch.object(planning,"urlopen",return_value=io.BytesIO(json.dumps(reply).encode())):
            assert planning.ask(cfg,{},["a"],scratch/"log1","task-x")[0]=="a"
        with patch.object(planning,"urlopen",side_effect=TimeoutError):
            try:
                planning.ask(cfg,{},["a"],scratch/"log2","task-x")
            except TimeoutError:
                pass
        with closing(budget.connect()) as db:
            rows=db.execute("select state,amount from calls order by id").fetchall()
            assert rows==[("settled","0.0012"),("reserved","10.0005")]
        # A new process cannot reset the remaining budget by choosing a fresh run log.
        command="from budget import reserve; from decimal import Decimal; import json,sys; reserve(sys.argv[1],Decimal(sys.argv[2]),json.loads(sys.argv[3]))"
        def attempt(key,amount):
            return subprocess.Popen([sys.executable,"-c",command,key,amount,json.dumps(cfg)],
                                    cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        assert attempt("task-x","10").wait()!=0
        racers=[attempt("task-y","12"),attempt("task-y","12")]
        assert sorted(p.wait()==0 for p in racers)==[False,True]
        print("PASS pairwise ordering and validation-only, group/time CLI, partition isolation, sample submission order and replay, timeout checkpoint export, mocked usage, restart and concurrent budget cap")


if __name__=="__main__":
    main()
