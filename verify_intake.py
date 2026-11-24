"""Exercise automatic intake using controlled fixtures and, optionally, the original course ZIP."""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "selection"))
from task_intake import generate, IntakeError


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def watch(source, allowed):
    """Independently reject any attempted data read outside the allowlist."""
    opened = []
    def check(name):
        if name not in allowed:
            raise AssertionError(f"Unexpected content read: {name}")
        opened.append(name)
    if source.is_dir():
        original = Path.open
        def file_open(path, *args, **kwargs):
            if path.is_relative_to(source):
                check(path.relative_to(source).as_posix())
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", file_open):
            yield opened
    else:
        original = ZipFile.open
        def zip_open(archive, entry, *args, **kwargs):
            check(getattr(entry, "filename", entry))
            return original(archive, entry, *args, **kwargs)
        with patch.object(ZipFile, "open", zip_open):
            yield opened


def csv_bytes(header, rows):
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return buf.getvalue()


def suite(course):
    cases = []
    def execute(name, source, out, allowed, expected, *, readme=None, trust=False, error=None, exact=False):
        out.mkdir()
        try:
            with watch(source, allowed) as opened:
                try:
                    generate(source, readme, out, trust)
                except IntakeError:
                    if error is None:
                        raise
            audit = read(out / "intake.json")
            recorded = [f["file"] for f in audit["opened"]]
            assert recorded == opened, (recorded, opened)
            if error is not None:
                assert audit["status"] == "needs_clarification" and error in audit["reason"], audit
                assert not (out / "task.json").exists()
            else:
                assert set(opened) == set(allowed)
                spec = read(out / "task.json")
                for k, v in expected.items():
                    assert spec.get(k) == v, (k, spec.get(k), v)
                if exact:
                    assert spec == expected
                from objective import METRICS
                status = "ready" if spec["metric"] in METRICS else "unsupported_training_metric"
                if spec.get("allow_pickle_features") is False and status=="ready":
                    status="requires_trusted_objects"
                assert audit["status"] == status
            cases.append(dict(case=name, passed=True, status=audit["status"], opened=opened))
        except Exception as exc:
            cases.append(dict(case=name, passed=False, error=f"{type(exc).__name__}: {exc}"))

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        def fixture(name, doc, files, allowed, expected=None, *, error=None, archive=False, extra_docs=None):
            home = base / name
            source = home / "data"
            source.mkdir(parents=True)
            members = {"README.md":doc.encode(), "validation.csv":b"held out", "test.csv":b"held out", **files}
            if extra_docs:
                members.update(extra_docs)
            for key, raw in members.items():
                path = source / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            if archive:
                source = home / "task.zip"
                with ZipFile(source,"w") as z:
                    for key, raw in members.items():
                        z.writestr(key, raw)
            execute(name, source, home/"report", allowed, expected or {}, error=error)

        allowed = {"README.md", "train.csv"}
        rows = [["A","yes","01",2],["B","no","02",3]]
        header = ["town","bought","region","visits"]
        data = {"train.csv":csv_bytes(header, rows)}
        features = {"target":"bought","features":["visits","town"],"numeric":["visits"],"metric":"accuracy"}
        desc = "The target column is `bought`. Use `visits` and `town` as inputs. "
        fixture("renamed-and-reordered",desc+"Report accuracy.",data,allowed,features)
        fixture("zip-directory-parity",desc+"Report accuracy.",data,allowed,features,archive=True)
        fixture("markdown-list","**Target column**: `bought`\n## Features\n- `visits`\n- `town`\n## Metric\n- accuracy",data,allowed,features)
        fixture("predict-using","Predict the `bought` column using `visits` and `town`. Report accuracy.",data,allowed,features)
        fixture("chinese","目标列为 `购买`。输入字段为 `次数`、`地区`。评分指标为准确率。",
                {"train.csv":csv_bytes(["地区","购买","次数"],[["东",1,7],["西",0,8]])},allowed,
                {"target":"购买","features":["次数","地区"],"numeric":["次数"]})
        fixture("implicit-inputs","Target: `bought`. Do not use `town` as input. Report accuracy.",data,allowed,
                {"features":["region","visits"],"numeric":["visits"]})
        fixture("nonstandard-training-name","The training file is `examples.csv`. "+desc+"Report accuracy.",
                {"examples.csv":data["train.csv"]},{"README.md","examples.csv"},features)
        fixture("split-contradiction","The training file is `test.csv`. "+desc+"Report accuracy.",data,
                {"README.md"},error="split role")
        fixture("absent-target",desc.replace("bought","renewed")+"Report accuracy.",data,allowed,error="absent")
        fixture("two-class-not-pair","Predict the `bought` column using `visits` and `town`. "
                "Decide which of the two classes applies. Report accuracy.",data,allowed,{"pairwise":None,"modality":"table"})
        fixture("secondary-metric",desc+"Primary metric: macro-F1. Also report accuracy.",data,allowed,{"metric":"macro_f1"})
        fixture("code-formatted-metric",desc+"Report `macro-F1`.",data,allowed,{"metric":"macro_f1"})
        fixture("secondary-alone",desc+"Secondary metric: accuracy.",data,allowed,error="metric")
        fixture("top-k-accuracy",desc+"Report top-5 accuracy.",data,allowed,{"metric":"top_5_accuracy"})
        fixture("backticked-top-k",desc+"Official metric: `top-3 accuracy`.",data,allowed,{"metric":"top_3_accuracy"})
        fixture("actual-metric-on-next-line",desc+"\n## Primary metric\nmacro-F1\nAlso report accuracy.",data,allowed,{"metric":"macro_f1"})
        fixture("compound-primary",desc+"Primary metric: balanced accuracy; also report accuracy.",data,allowed,{"metric":"balanced_accuracy"})
        fixture("negated-metric",desc+"Do not use accuracy. Evaluate using macro-F1.",data,allowed,{"metric":"macro_f1"})
        fixture("unrecognised-primary",desc+"Primary metric: custom contest points. Also report accuracy.",data,allowed,error="metric")
        fixture("missing-metric",desc,data,allowed,error="metric")
        fixture("ambiguous-metric",desc+"Report macro-F1 and accuracy.",data,allowed,error="metric")
        fixture("missing-column",desc.replace("visits","visit_count")+"Report accuracy.",data,allowed,error="missing")
        fixture("label-as-input",desc.replace("visits","bought")+"Report accuracy.",data,allowed,error="overlap")
        fixture("conflicting-target",desc+"The target is `visits`. Report accuracy.",data,allowed,error="target")
        fixture("duplicate-header",desc+"Report accuracy.",{"train.csv":csv_bytes(header[:3]+["bought"],rows)},allowed,error="header")
        fixture("ragged-csv",desc+"Report accuracy.",{"train.csv":csv_bytes(header,[rows[0],rows[1][:2]])},allowed,error="CSV")
        fixture("missing-label",desc+"Report accuracy.",{"train.csv":csv_bytes(header,[rows[0],["A","","04",3]])},allowed,error="labels")
        fixture("ambiguous-readme",desc,data,set(),error="README",extra_docs={"other/README.md":b"other"})
        # Mixed split formats would themselves be an error. Use .npy held-out fixtures here.
        arrays = {"inputs_train.npy":npy_bytes([[2,6],[3,7],[1,5]]), "answers_train.npy":npy_bytes([1,0,1])}
        doc = ("`inputs_*.npy` contains features. `answers_*.npy` contains labels. "
               "The columns from left to right are `age`, `visits`. Report accuracy.")
        def array_case(name, content, values, expected=None, error=None):
            home=base/name
            source=home/"data"
            source.mkdir(parents=True)
            (source/"README.md").write_text(content,encoding="utf-8")
            for key,raw in values.items():
                (source/key).write_bytes(raw)
            for split in ("test","validation"):
                for stem in ("inputs","answers"):
                    (source/f"{stem}_{split}.npy").write_bytes(b"held out")
            execute(name,source,home/"report",{"README.md",*values},expected or {},error=error)
        array_case("named-array-roles",doc,arrays,{"train_labels":"answers_train.npy","features":["age","visits"],"numeric":["age","visits"]})
        array_case("array-width",doc.replace(", `visits`",""),arrays,error="width")
        array_case("array-row-alignment",doc,{**arrays,"answers_train.npy":npy_bytes([1,0])},error="row counts")
        array_case("object-header-only",doc,{**arrays,"inputs_train.npy":npy_bytes(np.array([["a","b"]]*3, dtype=object))},
                   {"allow_pickle_features":False,"numeric":[]})
        if course:
            for path in sorted((ROOT/"tasks").glob("*.json")):
                ref=read(path)
                name=path.stem
                with ZipFile(course) as z:
                    member,=[p for p in z.namelist() if p.lower()==f"final_project_tasks/{name}/readme.md"]
                execute("course-"+name,course,base/("report-"+name),{member,ref["train"],*([ref["train_labels"]] if "train_labels" in ref else [])},
                        ref,readme=member,trust=True,exact=True)
    return cases


def replay_contracts(runs, source):
    checks = []
    for path in sorted(runs.glob("*/task.json")):
        run = path.parent
        name = "saved-contract-" + run.name
        try:
            contract = read(path)
            audit = read(run / "intake.json")
            result = read(run / "result.json")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            assert checksum == result["contract_sha256"], "training manifest digest differs"
            allowed = {audit["readme"], contract["train"]}
            if "train_labels" in contract:
                allowed.add(contract["train_labels"])
            with tempfile.TemporaryDirectory() as tmp, watch(source, allowed) as opened:
                regenerated = generate(source, audit["readme"], Path(tmp), contract.get("allow_pickle_features",False))
                assert read(regenerated) == contract, "regenerated task differs"
                assert read(regenerated.parent / "intake.json") == audit, "source evidence differs"
                assert set(opened) == allowed, "source read set differs"
            checks.append(dict(case=name, passed=True, status="reproduced", opened=opened, contract_sha256=checksum))
        except Exception as exc:
            checks.append(dict(case=name, passed=False, error=f"{type(exc).__name__}: {exc}"))
    if not checks:
        raise ValueError("No saved automatic task manifests in the specified runs directory")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs", type=Path, help="Additionally reproduce saved manifests and check the training contract")
    args = parser.parse_args()
    results = suite(args.course)
    if args.runs:
        if not args.course:
            parser.error("--runs requires --course")
        results.extend(replay_contracts(args.runs,args.course))
    passed = sum(r["passed"] for r in results)
    report = {"passed":passed,"total":len(results),"cases":results}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        if not result["passed"]:
            print(result["case"], result["error"])
    print(f"{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)
