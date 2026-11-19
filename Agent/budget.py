"""Durable per-task reservations shared by both training entry points."""
import hashlib
import os
import sqlite3
from decimal import Decimal
from pathlib import Path


def task_key(train_file, spec):
    if spec.get("budget_task_id"):
        return str(spec["budget_task_id"])
    with train_file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def connect():
    default = Path.home() / ".aiaa3102" / "budget.sqlite3"
    path = Path(os.environ.get("AIAA_BUDGET_DB", default))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS calls
        (id INTEGER PRIMARY KEY, task TEXT NOT NULL, amount TEXT NOT NULL,
         state TEXT NOT NULL, model TEXT NOT NULL, source TEXT NOT NULL,
         input_rate TEXT NOT NULL, output_rate TEXT NOT NULL,
         prompt_tokens INTEGER, completion_tokens INTEGER, request_id TEXT)""")
    return db


def reserve(task, amount, cfg):
    db = connect()
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            total = sum(Decimal(row[0]) for row in
                        db.execute("SELECT amount FROM calls WHERE task=?", (task,)))
            if total + amount > 20:
                raise ValueError("Task budget would exceed RMB 20")
            row = db.execute("""INSERT INTO calls
                (task,amount,state,model,source,input_rate,output_rate) VALUES (?,?,?,?,?,?,?)""",
                (task,str(amount),"reserved",cfg["model"],cfg["price_source"],
                 str(cfg["input_rmb_per_million"]),str(cfg["output_rmb_per_million"])))
            return row.lastrowid
    finally:
        db.close()


def settle(call, amount, n, m, request_id):
    db = connect()
    try:
        with db:
            cursor = db.execute("""UPDATE calls SET amount=?,state='settled',
              prompt_tokens=?,completion_tokens=?,request_id=? WHERE id=? AND state='reserved'""",
              (str(amount), n, m, request_id, call))
            if cursor.rowcount != 1:
                raise RuntimeError("Reservation missing or already settled")
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Export all recorded calls for a task")
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.task.read_text(encoding="utf-8-sig"))
    key = task_key(args.task.parent / spec["train"], spec)
    db = connect()
    try:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM calls WHERE task=? ORDER BY id", (key,))]
    finally:
        db.close()
    print(json.dumps(dict(task=key,calls=rows,committed_rmb=str(sum(
        (Decimal(r["amount"]) for r in rows), Decimal(0)))),ensure_ascii=False,indent=2))
