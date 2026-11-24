"""Exercise the integrated CLI contracts using synthetic data only."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Agent"))
sys.path.append(str(ROOT/"experiments/selection"))
from agent import inputs, predict
from develop import local_decision
from features import build
from models import save_bundle
from task_data import data_file, data_root
from validation import split_rows


def check_pairs():
    # Include transitive connections, reversed pairs, and capitalization changes.
    rows = []
    for i in range(12):
        a, b, c = [f"{x} {i}" for x in ("alpha", "beta", "gamma")]
        rows.extend([(a,b,"0"),(b.upper(),a,"1"),(c,b,"0")])
    df = pd.DataFrame(rows, columns=["left","right","label"])
    spec = dict(target="label", features=["left","right"], metric="accuracy", pairwise=True)
    train, val = split_rows(df,spec,113)
    def titles(part):
        return {t.casefold() for t in part[spec["features"]].to_numpy().ravel()}
    assert not titles(train).intersection(titles(val))
    assert len(train)+len(val)==len(df)


def check_feedback():
    options = {name: name for name in ("pair_tfidf", "framing_rank", "contrast_rank")}
    # Hold the overall accuracy and candidate budget fixed. Reverse only which
    # slice contains the mistakes and verify that the next experiment changes.
    def choose(a, b, rows, candidates=options):
        slices = {"similar_wording": {"rows": rows, "correct": a},
                  "different_wording": {"rows": rows, "correct": b}}
        recent = dict(slices=slices, score=(a+b)/(2*rows), metric="accuracy", action="pair_hybrid")
        return local_decision("pair", candidates, recent)[0]
    assert choose(50, 90, 100) == "contrast_rank"
    assert choose(90, 50, 100) == "framing_rank"
    assert choose(1, 2, 2) == "pair_tfidf"
    assert choose(70, 70, 100) == "pair_tfidf"
    assert choose(50, 90, 100, {"pair_tfidf": "lexical baseline"}) == "pair_tfidf"


def check_models(tmp):
    x = pd.DataFrame({"x":[-3,-2,-1,1,2,3]})
    y = np.array(["0"]*3+["1"]*3)
    model = LogisticRegression().fit(x,y)
    spec = dict(train="unused.csv", test="synthetic-test.csv", target="label", features=["x"],
                numeric=["x"], modality="table", metric="accuracy", output="label")
    schema = dict(features=["x"],numeric=["x"],mode="table")
    csv = x.to_csv(index=False).encode()
    array = io.BytesIO()
    np.save(array,x.to_numpy())
    archive = tmp/"examples.zip"
    with ZipFile(archive,"w") as out:
        out.writestr("x.csv",csv)
        out.writestr("x.npy",array.getvalue())
        out.writestr("synthetic-test.csv",x.assign(label=y).to_csv(index=False))
    old = dict(classes=model.classes_, weights={"m":1.}, models={"m":model}, schema=schema, spec=spec)
    legacy = tmp/"legacy.joblib"
    joblib.dump(old,legacy)
    modern = tmp/"model.joblib"
    save_bundle(modern,spec,model.classes_,{"m":model},"full_refit")
    cli = [sys.executable,"-B",str(ROOT/"Agent/agent.py")]
    for bundle in (legacy,modern):
        for ext in ("csv","npy"):
            output=tmp/f"{bundle.stem}-{ext}.csv"
            subprocess.run([*cli,"predict","--data",str(archive),"--model",str(bundle),
                            "--input","x."+ext,"--output",str(output)],check=True)
            frame=pd.read_csv(output,dtype=str)
            assert np.array_equal(frame.iloc[:,0],y)
        assert (tmp/f"{bundle.stem}-csv.csv").read_bytes()==(tmp/f"{bundle.stem}-npy.csv").read_bytes()

    invalid=tmp/"invalid.csv"
    x.assign(label=[*y[:-1],"unknown"]).to_csv(invalid,index=False)
    dest=tmp/"invalid-predictions.csv"
    try:
        predict({**schema,"model":model,"spec":spec},invalid,dest,score_test=True)
    except ValueError as err:
        assert "unknown class" in str(err) and not dest.exists()
    else:
        raise AssertionError("Invalid labels were accepted")

    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    task=tmp/"task.json"
    task.write_text(json.dumps(spec),encoding="utf-8")
    nomination=tmp/"challenger.json"
    nomination.write_text("{}",encoding="utf-8")
    result=dict(selected=["m"],stage="full_refit",model_sha256=fingerprint(modern),
                contract_sha256=fingerprint(task),challenger_lock_sha256=fingerprint(nomination))
    (tmp/"result.json").write_text(json.dumps(result),encoding="utf-8")
    subprocess.run([*cli,"validate","--split","test","--data",str(archive),"--out",str(tmp)],check=True)
    scored=json.loads((tmp/"result.json").read_text(encoding="utf-8"))
    assert scored["test_correct"]==scored["test_rows"]==len(y)
    assert (tmp/"test-predictions.csv").exists() and "validation_score" not in scored


def check_legacy_images(tmp):
    folder = tmp/"images"
    folder.mkdir()
    names = ["dark.png", "light.png"]
    for name, value in zip(names, (12, 240)):
        Image.new("L", (8, 8), value).save(folder/name)
    frame = pd.DataFrame({"image": names})
    frame.to_csv(folder/"rows.csv", index=False)
    archive = tmp/"image-input.zip"
    with ZipFile(archive, "w") as z:
        for path in folder.iterdir():
            z.write(path, "nested/"+path.name)
    schema = dict(mode="image", features=["image"], numeric=[])
    x = inputs(frame, schema, folder)
    model = build("image", "color", x, 113).fit(x, ["0","1"])
    spec = dict(target="label", features=["image"], output="label", metric="accuracy")
    bundle = dict(**schema, model=model, spec=spec)
    expected, actual = tmp/"image-dir.csv", tmp/"image-zip.csv"
    predict(bundle, folder/"rows.csv", expected)
    with data_root(archive) as root:
        predict(bundle, data_file(root,"nested/rows.csv"), actual)
    assert expected.read_bytes() == actual.read_bytes()
    assert pd.read_csv(actual, dtype=str)["label"].tolist() == ["0","1"]


if __name__=="__main__":
    check_pairs()
    check_feedback()
    with TemporaryDirectory() as folder:
        check_models(Path(folder))
        check_legacy_images(Path(folder))
    print("PASS feedback-dependent choices, grouped pairs, ZIP cross-format replay, failed scoring and locked final export")
