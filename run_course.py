"""Run the Agent on course materials with process-wide read auditing and separate stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FILES = {
    "headline": ("final_project_tasks/headlines_ctr_clean", "README.md", ["train.csv"], ["val.csv"]),
    "coupon": ("final_project_tasks/in_vehicle_coupon_recommendation", "Readme.md", ["C_train.npy","y_train.npy"], ["C_val.npy","y_val.npy"])
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train","validate"])
    parser.add_argument("task", choices=FILES)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT/"runs")
    args = parser.parse_args()
    args.runs.mkdir(parents=True, exist_ok=True)
    out = args.runs/args.task
    base, readme, train, val = FILES[args.task]
    training = args.stage == "train"
    if training and out.exists():
        raise FileExistsError(out)
    if not training and ((out/"validation-predictions.csv").exists() or not (out/"result.json").is_file()):
        raise ValueError("Validation requires a locked run with no prior validation predictions")
    audit = args.runs/f"{args.task}-{args.stage}-access.jsonl"
    if audit.exists():
        raise FileExistsError(audit)
    members = [base+"/"+name for name in ([readme,*train] if training else val)]
    env = dict(os.environ, AIAA_STUDY_ZIP=str(args.data.resolve()), AIAA_STUDY_MEMBERS=json.dumps(members),
               AIAA_STUDY_ACCESS=str(audit.resolve()), PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT/"checks"),env.get("PYTHONPATH","")])
    common=["--data",str(args.data.resolve()),"--out",str(out.resolve())]
    if training:
        command=["develop",*common,"--readme",base+"/"+readme,"--provider","local","--strengthen",
                 "--max-actions","3","--train-only","--seconds","1200"]
        if args.task=="coupon":
            command.append("--trust-object-features")
    else:
        command=["validate",*common]
    subprocess.run([sys.executable,"-B",str(ROOT/"Agent/agent.py"),*command],env=env,check=True)
    records=[json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    if {item["file"] for item in records} != set(members) or not all(item["allowed"] for item in records):
        raise RuntimeError("Unexpected course archive access")


if __name__=="__main__":
    main()
