"""Coverage audit: the curated master vs the full Sononym/Fourier library.

Finds blindspots three ways:
  1. Taxonomy   - Sononym categories with library samples no curated category covers.
  2. CLAP       - library samples far (low cosine) from every curated sample; the
                  poorly-covered ones are clustered and named by the local LLM.
  3. Provenance - packs well represented in the library but thin in the curated set.

`run_audit()` returns a summary dict and optionally writes a markdown report.
Exposed as `fourier tools audit`.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .curate import load_index, _pack_of
from .curate_config import CATEGORIES

DEFAULT_MODEL = "qwen3.5:9b"
_MACHINES = {"808", "909", "606", "707", "101", "303", "727", "555"}


def _toks(fn):
    return [t for t in re.split(r"[^a-z0-9]+", (fn or "").lower())
            if (len(t) >= 2 and not t.isdigit()) or t in _MACHINES]


def _llm_label(profiles, model):
    prof = "\n".join(profiles)
    try:
        payload = {"model": model, "think": False, "stream": False, "format": "json",
                   "options": {"temperature": 0.2},
                   "prompt": "Each item is a cluster of samples our curated pack UNDER-represents. "
                             "Give a short 3-7 word label naming what KIND of sound each is, from its "
                             "categories, pack names and filename words. Invent nothing.\n" + prof +
                             '\n\nReply ONLY JSON: {"d": {"0": "label", ...}}'}
        req = urllib.request.Request("http://localhost:11434/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        from ..net import local_opener                  # localhost: never through a proxy
        return json.loads(json.loads(local_opener().open(req, timeout=300).read())
                          .get("response", "{}")).get("d", {})
    except Exception:
        return {}


def run_audit(session, master_dir, model=DEFAULT_MODEL, far=0.55, out_path=None, log=print):
    from sklearn.cluster import KMeans

    master = Path(master_dir)
    lines = []
    def w(s=""):
        log(s); lines.append(s)

    ids, emb = load_index()
    ids = ids.astype(np.int64); emb = emb.astype("float32")
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    id2row = {int(i): r for r, i in enumerate(ids)}

    meta = {}
    from ..metadata.rows import fetch, sample_select
    for sid, path, rel, fn, cats, cls in fetch(
            session, sample_select("id", "path", "rel_path", "filename", "categories", "classes",
                                   session=session)):
        meta[sid] = dict(pack=_pack_of(rel, path or ""), fn=fn or "", cats=list(cats or ()))
    w(f"# Curated vs library coverage audit\n\nLibrary: {len(meta):,} samples, "
      f"{len(ids):,} with CLAP embeddings.\n")

    # map curated master files -> library ids by filename stem
    cur_stems = {os.path.splitext(p.name)[0].lower()
                 for p in master.rglob("*") if p.suffix.lower() == ".wav" and not p.name.startswith("_")}
    stem2ids = defaultdict(list)
    for sid, m in meta.items():
        stem2ids[os.path.splitext(m["fn"])[0].lower()].append(sid)
    cur_ids = {i for st in cur_stems for i in stem2ids.get(st, [])}
    cur_rows = sorted({id2row[i] for i in cur_ids if i in id2row})
    if not cur_rows:
        raise RuntimeError("no curated files mapped to the library; is the master built?")
    C = emb[np.array(cur_rows)]
    w(f"Curated master mapped to {len(cur_rows):,} library samples (by filename stem).\n")

    # nearest-curated cosine for every embedded sample
    nn = np.empty(len(ids), dtype="float32")
    for i in range(0, len(ids), 4096):
        nn[i:i + 4096] = (emb[i:i + 4096] @ C.T).max(axis=1)
    row2id = {r: int(i) for i, r in id2row.items()}
    poor = int((nn < far).sum())
    w(f"## CLAP coverage\n\nNearest-curated cosine: median {np.median(nn):.2f}, mean {nn.mean():.2f}. "
      f"{poor:,} samples ({poor/len(ids)*100:.1f}%) are below {far:.2f} (potential blindspots).\n")

    # 1) taxonomy coverage
    # a classifier category is covered when its canonical label is one a category accepts
    from ..metadata.vocab import canonical
    accepted = {lab for c in CATEGORIES.values() for lab in (c.get("labels") or ())}
    catcount = Counter(); cat_cov = defaultdict(list)
    for sid, m in meta.items():
        for part in m["cats"]:
            catcount[part] += 1
            r = id2row.get(sid)
            if r is not None:
                cat_cov[part].append(nn[r])
    covered = lambda cat: canonical("sononym", "category", cat) in accepted
    uncovered = sorted([(c, n) for c, n in catcount.items() if not covered(c) and n >= 150],
                       key=lambda x: -x[1])
    w("## 1. Taxonomy coverage\n")
    w("| Category | Library | Covered? | Median cos |")
    w("| --- | --- | --- | --- |")
    for cat, n in catcount.most_common():
        if n < 150:
            continue
        med = float(np.median(cat_cov[cat])) if cat_cov[cat] else float("nan")
        w(f"| {cat} | {n:,} | {'yes' if covered(cat) else 'NO'} | {med:.2f} |")
    w(f"\nUncovered categories (>=150 samples): "
      f"{', '.join(f'{c} ({n:,})' for c, n in uncovered) or 'none'}\n")

    # 3) provenance coverage
    pack_lib = Counter(); pack_cur = Counter()
    for sid, m in meta.items():
        pack_lib[m["pack"]] += 1
        if sid in cur_ids:
            pack_cur[m["pack"]] += 1
    w("## 3. Provenance coverage (big packs, thin curation)\n")
    w("| Pack | Library | Curated |")
    w("| --- | --- | --- |")
    big = [(p, n, pack_cur.get(p, 0)) for p, n in pack_lib.most_common() if n >= 400]
    for p, n, c in sorted(big, key=lambda x: (x[2] / max(1, x[1]), -x[1]))[:15]:
        w(f"| {p} | {n:,} | {c} |")
    w("")

    # 2) CLAP blindspot clusters, named by the local LLM
    poor_rows = np.where(nn < far)[0]
    w(f"## 2. CLAP blindspot clusters ({len(poor_rows):,} poorly-covered)\n")
    clusters = []
    if len(poor_rows) >= 50:
        k = min(15, max(5, len(poor_rows) // 400))
        lab = KMeans(n_clusters=k, random_state=0, n_init=4).fit_predict(emb[poor_rows])
        for c in range(k):
            rws = poor_rows[lab == c]
            sids = [row2id[int(r)] for r in rws if int(r) in row2id]
            cats = Counter(p for s in sids for p in meta[s]["cats"])
            packs = Counter(meta[s]["pack"] for s in sids)
            tk = Counter(t for s in sids for t in _toks(meta[s]["fn"]))
            clusters.append(dict(id=c, n=len(sids), cats=cats.most_common(3),
                                 packs=packs.most_common(3), toks=[t for t, _ in tk.most_common(8)],
                                 ex=[meta[s]["fn"] for s in sids[:6]], avgcos=float(nn[rws].mean())))
        names = _llm_label(
            ["%d: categories=%s; packs=%s; words=%s; examples=%s" % (
                c["id"], ", ".join(f"{a}({b})" for a, b in c["cats"]),
                ", ".join(f"{a}({b})" for a, b in c["packs"]),
                ", ".join(c["toks"]), " | ".join(c["ex"])) for c in clusters], model)
        for c in sorted(clusters, key=lambda x: -x["n"]):
            label = names.get(str(c["id"]), "") or f"cluster {c['id']}"
            w(f"- **{label}** - {c['n']:,} samples, avg cos {c['avgcos']:.2f}. "
              f"categories: {', '.join(a for a, _ in c['cats'])}; "
              f"packs: {', '.join(a for a, _ in c['packs'])}; words: {', '.join(c['toks'][:6])}")

    if out_path:
        Path(out_path).write_text("\n".join(lines))
        log(f"\nreport written: {out_path}")
    return dict(poorly_covered=poor, uncovered_categories=uncovered,
                median_cos=float(np.median(nn)), curated_mapped=len(cur_rows),
                blindspot_clusters=[(c["id"], c["n"]) for c in clusters])
