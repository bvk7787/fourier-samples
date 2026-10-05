"""Resumable --all builds: what a build into <master>.next has finished, and whether it can
be picked up again.

Each category that finishes writes its summary and file entries to
<master>.next/.progress/<CATEGORY>.json. `fourier build --resume` keeps those
categories when nothing a build depends on has changed since (the fingerprint: the code, every
tunable, the ratings, the seed, the providers, the database's samples and features, the CLAP
index and the build options) and builds only the rest; the result is the same as a build
that never stopped, because every category is built independently and seeded. Anything else
starts the build over. The .progress folder is removed before the build is checked and
synced into the master.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

PROGRESS_DIR = ".progress"
_SRC = Path(__file__).resolve().parents[1]          # src/fourier


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(_SRC.rglob("*.py")):
        h.update(str(p.relative_to(_SRC)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _db_state(session) -> list:
    from sqlalchemy import text
    out = []
    for sql in ("SELECT COUNT(*), MAX(id) FROM samples",
                "SELECT COUNT(*) FROM sample_features",
                "SELECT COUNT(*) FROM labels", "SELECT COUNT(*) FROM descriptors",
                # samples a walk marked missing (or found again) change what a build picks
                "SELECT COUNT(*) FROM missing_files"):
        try:
            out.append([None if v is None else int(v) for v in session.execute(text(sql)).first()])
        except Exception:
            out.append(None)
    return out


def fingerprint(session, categories, opts: dict) -> str:
    """What a build's output depends on, hashed. Two builds with the same fingerprint make
    the same categories."""
    from ..paths import clap_index_path
    from .curate import _build_meta
    meta = _build_meta()
    meta.pop("git_sha", None)                    # the code itself is hashed below
    meta.pop("fourier_version", None)
    idx = clap_index_path()
    try:
        st = idx.stat()
        index = [st.st_size, int(st.st_mtime)]
    except OSError:
        index = None
    doc = dict(meta=meta, code=_code_hash(), db=_db_state(session), index=index,
               categories=sorted(categories), opts={k: opts[k] for k in sorted(opts)},
               env={k: os.environ.get(k) for k in ("FOURIER_NO_STICKY",)})
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:24]


class Progress:
    """The finished categories of one build into build_dir."""

    def __init__(self, build_dir, fp: str):
        self.build_dir = Path(build_dir)
        self.dir = self.build_dir / PROGRESS_DIR
        self.fp = fp

    def start(self, resume: bool, log=print) -> dict:
        """{category: (summary, entries)} already built, when resuming a build with the same
        fingerprint; else {} and an empty build_dir."""
        done = {}
        if resume and (self.dir / "fingerprint").is_file():
            if (self.dir / "fingerprint").read_text().strip() == self.fp:
                for p in sorted(self.dir.glob("*.json")):
                    cat = p.stem
                    try:
                        d = json.loads(p.read_text())
                    except (OSError, ValueError):
                        continue
                    if (self.build_dir / cat).is_dir():
                        done[cat] = (d["summary"], d["entries"])
                log(f"resuming: {len(done)} categories already built "
                    f"({', '.join(sorted(done)) or 'none'})")
                return done
            log("not resuming: the code, settings, ratings or library changed since that build; "
                "starting over")
        elif resume:
            log(f"nothing to resume in {self.build_dir}; building everything")
        if self.build_dir.exists():
            if any(self.build_dir.iterdir()) and not (self.dir.exists()
                                                      or (self.build_dir / "manifest.json").exists()):
                raise RuntimeError(f"{self.build_dir} exists and isn't a Fourier build: move it aside")
            shutil.rmtree(self.build_dir)
        self.dir.mkdir(parents=True)
        (self.dir / "fingerprint").write_text(self.fp + "\n")
        return done

    def done(self, category: str, summary: dict, entries: list) -> None:
        tmp = self.dir / f".{category}.tmp"
        tmp.write_text(json.dumps({"summary": summary, "entries": entries},
                                  default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        os.replace(tmp, self.dir / f"{category}.json")

    def finish(self) -> None:
        """The build is whole: drop the progress, so it never reaches the master."""
        shutil.rmtree(self.dir, ignore_errors=True)


def resumable(build_dir) -> bool:
    return (Path(build_dir) / PROGRESS_DIR / "fingerprint").is_file()
