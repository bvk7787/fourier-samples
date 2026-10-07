"""Reading and writing a master's manifest.json with library-relative source paths.

From manifest format 3 an entry's `src` (in `categories` and `sets`) is stored relative to the
library folder it sits in, and the manifest lists those folders once, under `src_roots`, with
the home folder written as `~`. An entry under a folder other than the first adds `src_root`
(its index). A manifest then names no one's home folder in its thousands of entries and reads
the same wherever the library is mounted. A source outside every library folder (or under one
spelled differently, a symlink or backslashes) keeps its absolute path, as before.

Every reader goes through read(), which turns each stored `src` back into the absolute path
the build used, so the rest of Fourier sees the manifest it always did; format 1 and 2
manifests (absolute paths throughout) read unchanged. write() stores the relative form, and
leaves the caller's document as it was.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FORMAT = 3          # 2 added each entry's canonical labels; 3 the library-relative `src`
SECTIONS = ("categories", "sets")


def _home() -> str | None:
    h = os.path.expanduser("~")
    return h if h not in ("", "/", "~") else None


def collapse_home(path: str) -> str:
    """The path with the home folder written as ~ (unchanged outside it)."""
    h = _home()
    return "~" + path[len(h):] if h and path.startswith(h + "/") else path


def _expand(root: str) -> str:
    return os.path.expanduser(root) if root.startswith("~/") else root


def _is_relative(src: str) -> bool:
    return not (src.startswith(("/", "~", "\\")) or (len(src) > 1 and src[1] == ":"))


def _entries(doc: dict):
    for sect in SECTIONS:
        for cd in (doc.get(sect) or {}).values():
            if isinstance(cd, dict):
                yield from (e for e in cd.get("entries") or () if isinstance(e, dict))


def resolve(doc: dict) -> dict:
    """The document with every stored `src` made absolute again (in place, and returned)."""
    if not isinstance(doc, dict):
        return doc
    roots = [_expand(r) for r in doc.pop("src_roots", None) or ()]
    for e in _entries(doc):
        i = e.pop("src_root", 0)
        src = e.get("src")
        if roots and isinstance(src, str) and src and _is_relative(src) and 0 <= i < len(roots):
            e["src"] = f"{roots[i]}/{src}"
    return doc


def stored(doc: dict) -> dict:
    """A copy of the document in its stored form: each `src` under a library folder relative to
    it, the folders listed once (`src_roots`). The caller's document is left as it is."""
    from ..places import library_root_of
    if doc.get("src_roots"):                     # already stored: from its absolute form
        doc = resolve(json.loads(json.dumps(doc)))
    out = dict(doc)
    out.pop("src_roots", None)
    found: list[str] = []                       # roots, in the order the entries meet them
    rows = []                                   # (entry copy, root or None, relative path)
    for sect in SECTIONS:
        if not isinstance(doc.get(sect), dict):
            continue
        out[sect] = {}
        for name, cd in doc[sect].items():
            if not isinstance(cd, dict) or not isinstance(cd.get("entries"), list):
                out[sect][name] = cd
                continue
            ents = []
            for e in cd["entries"]:
                if not isinstance(e, dict):
                    ents.append(e)
                    continue
                e = dict(e)
                e.pop("src_root", None)
                src = e.get("src")
                root: str | None = None
                if isinstance(src, str) and src and not _is_relative(src):
                    root = next((r for r in found if src.startswith(r + "/")), None)
                    if root is None:
                        r = library_root_of(src)
                        if r and src.startswith(r.rstrip("/") + "/"):
                            root = r.rstrip("/")
                            found.append(root)
                rows.append((e, root, src[len(root) + 1:] if root and isinstance(src, str) else None))
                ents.append(e)
            out[sect][name] = dict(cd, entries=ents)
    if not found:
        return out
    order = sorted(set(found))
    index = {r: i for i, r in enumerate(order)}
    for e, root, rel in rows:
        if root is not None:
            e["src"] = rel
            if index[root]:
                e["src_root"] = index[root]
    out["src_roots"] = [collapse_home(r) for r in order]
    return out


def read(where) -> dict:
    """A manifest (a master folder or a manifest .json), its sources absolute."""
    p = Path(os.path.expanduser(str(where)))
    if p.is_dir():
        p = p / "manifest.json"
    return resolve(json.loads(p.read_text()))


def write(where, doc: dict) -> None:
    """Write a manifest (to a master folder or a .json path) in its stored form, atomically."""
    p = Path(os.path.expanduser(str(where)))
    if p.is_dir():
        p = p / "manifest.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(stored(doc), indent=2))
    os.replace(tmp, p)
