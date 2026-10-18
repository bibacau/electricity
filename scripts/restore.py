"""Restore a SQLite backup while the workbench is stopped. Preserve the old DB."""
import argparse
import shutil
import sqlite3
from pathlib import Path
from uuid import uuid4

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("backup",type=Path)
parser.add_argument("--data",type=Path,default=Path("data"))
parser.add_argument("--server-stopped",action="store_true",required=True)
args=parser.parse_args()
source=args.backup.resolve()
with sqlite3.connect(source.as_uri()+"?mode=ro",uri=True) as db:
    if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":
        raise SystemExit("Backup failed integrity check")
    required={"settings","conversations","messages","memories","feedback","events","experiments","checkpoints"}
    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not required.issubset(tables):
        raise SystemExit("Backup is not a workbench database")
args.data.mkdir(parents=True,exist_ok=True)
target=args.data/"workbench.sqlite3"
if source==target.resolve():
    raise SystemExit("Source must differ from destination")
if target.exists():
    shutil.copy2(target,args.data/f"before-restore-{uuid4().hex}.sqlite3")
temporary=args.data/f"restore-{uuid4().hex}.sqlite3"
shutil.copy2(source,temporary)
temporary.replace(target)
print("Database restored. Documents and model files are backed up separately.")
