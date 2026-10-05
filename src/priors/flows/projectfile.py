"""Project files: download a project and upload it later to continue (hosted apps may lose their disk).

A project file is a zip of the project folder (session state, evidence cards, registrations, the trial ledger,
the retrieval log) plus a small manifest. On upload, the file is checked before it is used:

* only plain relative paths and the file types a project contains are accepted;
* the total size is limited;
* every pre-registration record's hash and the trial ledger's hash chain are verified, so a project whose
  records were edited by hand is rejected.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ..registry import Registry

MANIFEST = "priors_project.json"
ALLOWED_SUFFIXES = {".json", ".jsonl"}
MAX_TOTAL_BYTES = 20 * 1024 * 1024
FORMAT = 1


class ProjectFileError(ValueError):
    pass


def export_project(root: str | Path, student: str = "", project: str = "") -> bytes:
    root = Path(root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps({
            "format": FORMAT, "student": student, "project": project,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }))
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in ALLOWED_SUFFIXES:
                z.write(f, f.relative_to(root).as_posix())
    return buf.getvalue()


def _safe(name: str) -> bool:
    p = PurePosixPath(name)
    return (not p.is_absolute() and ".." not in p.parts and ":" not in name and "\\" not in name
            and (p.suffix in ALLOWED_SUFFIXES or name.endswith("/")))


def read_manifest(data: bytes) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return json.loads(z.read(MANIFEST))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise ProjectFileError("This is not a Priors project file.") from e


def import_project(data: bytes, dest: str | Path, overwrite: bool = False) -> Path:
    """Unpack a project file into ``dest`` after checking it. Returns ``dest``."""
    dest = Path(dest)
    manifest = read_manifest(data)
    if manifest.get("format") != FORMAT:
        raise ProjectFileError("This project file comes from an incompatible version of Priors.")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if i.filename != MANIFEST]
        bad = [i.filename for i in infos if not _safe(i.filename)]
        if bad:
            raise ProjectFileError(f"The project file contains entries that are not allowed: {bad[:3]}")
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise ProjectFileError("The project file is too large.")
        if dest.exists() and any(dest.iterdir()):
            if not overwrite:
                raise ProjectFileError("A project with this name already exists here.")
            shutil.rmtree(dest)
        tmp = dest.with_name(dest.name + ".importing")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for i in infos:
            if i.is_dir():
                continue
            target = tmp / PurePosixPath(i.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(i))
    try:
        reg = Registry(tmp, require_evidence=False)
        reg.ledger.verify()
        reg.registrations()            # verifies every record's hash
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ProjectFileError(f"The project's records do not verify, so it cannot be used: {e}") from e
    tmp.rename(dest)
    return dest
