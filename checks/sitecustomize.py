"""Instrument reads from the course ZIP in the Agent and all of its Python children."""
import json
import os
from pathlib import Path
from zipfile import ZipFile

if os.environ.get("AIAA_STUDY_ZIP"):
    archive_path = Path(os.environ["AIAA_STUDY_ZIP"]).resolve()
    allowed = set(json.loads(os.environ["AIAA_STUDY_MEMBERS"]))
    audit = Path(os.environ["AIAA_STUDY_ACCESS"])
    original_open = ZipFile.open

    def checked_open(self, name, *args, **kwargs):
        if isinstance(self.filename, (str, os.PathLike)) and Path(self.filename).resolve() == archive_path:
            member = getattr(name, "filename", name)
            with audit.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid":os.getpid(), "file":member, "allowed":member in allowed})+"\n")
            if member not in allowed:
                raise PermissionError("Outside this stage's allowed course files: "+member)
        return original_open(self, name, *args, **kwargs)

    ZipFile.open = checked_open
