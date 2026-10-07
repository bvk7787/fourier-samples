"""A build's report: one HTML page to listen to the master and understand it, on any system,
without Live.

`fourier open report` writes it (into `<home>/reports`, never into the master) and opens it
in the web browser. Every category, its families and their files, each with a play button
(the audio is the master's own files, by file:// link), where it came from in the library,
a loop's tempo, and, per category, what the last build left out and why (the why log). Keep /
Drop / Misfiled buttons remember choices in the browser and export a CSV that `fourier review
import` reads, so rating needs neither Live nor typing names.
"""
from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path

DERIVED = ("KITS", "SLICE")


def _loops(master: Path) -> dict[str, dict]:
    """loops.csv by master path (DRUMLOOPS/<folder>/<file>)."""
    out = {}
    try:
        with open(master / "loops.csv", newline="") as f:
            for row in csv.DictReader(f):
                out[f"DRUMLOOPS/{row.get('folder')}/{row.get('file')}"] = row
    except OSError:
        pass
    return out


def _left_out(master: Path, cat: str) -> str:
    """What the last build left out of a category and why, in words (the why log)."""
    from .why_log import FILTER_WORDS, read
    doc = read(str(master), cat) or {}
    counts = {k: v for k, v in (doc.get("counts") or {}).items()
              if v and k in FILTER_WORDS and not k.endswith("readmitted")}   # the reasons
    found = doc.get("found")
    parts = []
    if found is not None:
        kept = doc.get("kept")
        parts.append(f"{found:,} candidate{'' if found == 1 else 's'} found"
                     + (f", {kept:,} kept" if kept is not None else ""))
    if counts:
        parts.append("left out: " + ", ".join(
            f"{n:,} {FILTER_WORDS.get(k, k.replace('_', ' ')).format(cat=cat)}"
            for k, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    if doc.get("status") == "empty":
        parts.append("left empty: too few samples the rules recognize")
    return "; ".join(parts)


def build_report(master_dir, out_path) -> dict:
    """Write the report for master_dir to out_path. Returns {files, categories}."""
    from ..places import library_rel
    master = Path(master_dir)
    from . import manifests
    man = manifests.read(master / "manifest.json")
    loops = _loops(master)
    cats = man.get("categories") or {}
    sets = man.get("sets") or {}
    esc = html.escape
    rows, toc, n_files = [], [], 0

    def file_row(cat: str, e: dict) -> str:
        rel = f"{cat}/{e['out']}"
        path = master / cat / e["out"]
        src = e.get("came_from") or ""
        frm = library_rel(e.get("src") or "") if e.get("src") else ""
        loop = loops.get(rel)
        tempo = f"{float(loop['bpm']):g} BPM" if loop and loop.get("bpm") else ""
        bars = f", {float(loop['bars']):g} bars" if loop and loop.get("bars") else ""
        name = os.path.basename(e["out"])
        extra = (f'<span class="meta">{esc(tempo + bars)}</span>' if tempo else "")
        origin = (f'<span class="from">from {esc(frm)}</span>' if frm else
                  f'<span class="from">a copy of {esc(src)}</span>' if src else "")
        rate = "" if cat in DERIVED else (
            '<span class="rate">'
            '<button data-r="keep" title="Keep: the next build keeps it">K</button>'
            '<button data-r="drop" title="Drop: the next build leaves it out">D</button>'
            '<button data-r="misfiled" title="Misfiled: it belongs in another category">M</button>'
            '</span>')
        why_cmd = esc('fourier why "' + rel + '"')
        return (f'<li data-path="{esc(rel)}" data-cat="{esc(cat)}">'
                f'<button class="play" data-src="{esc(path.as_uri())}" title="Play">&#9654;</button> '
                f'<span class="name" title="{why_cmd}: why it is here">{esc(name)}</span> '
                f'{extra} {origin} {rate}</li>')

    def section(cat: str, entries: list, note: str) -> str:
        fams: dict[str, list] = {}
        for e in entries:
            fams.setdefault(e.get("family") or os.path.dirname(e["out"]) or "-", []).append(e)
        body = "".join(
            f'<details><summary>{esc(f)} <span class="n">{len(es)}</span></summary><ul>'
            + "".join(file_row(cat, e) for e in es) + "</ul></details>"
            for f, es in fams.items())
        return (f'<section id="{esc(cat)}"><h2>{esc(cat)} <span class="n">{len(entries):,} files, '
                f'{len(fams)} famil{"y" if len(fams) == 1 else "ies"}</span></h2>'
                + (f'<p class="note">{esc(note)}</p>' if note else "") + body + "</section>")

    from .curate_config import CATEGORY_ORDER
    order = list(CATEGORY_ORDER)
    for cat, d in sorted(cats.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99):
        entries = (d or {}).get("entries") or []
        n_files += len(entries)
        toc.append(f'<a href="#{esc(cat)}">{esc(cat)} <span class="n">{len(entries):,}</span></a>')
        rows.append(section(cat, entries, _left_out(master, cat)))
    for s in DERIVED:
        entries = ((sets.get(s) or {}).get("entries")) or []
        if entries:
            what = ("drum kits drawn from the categories: a kick, snare, hats and more that go together"
                    if s == "KITS" else "drum loops cut to an even length, ready for slicing")
            toc.append(f'<a href="#{s}">{s} <span class="n">{len(entries):,}</span></a>')
            rows.append(section(s, entries, what))
    built = man.get("generated") or ""
    page = TEMPLATE.format(
        title=esc(f"Fourier build report: {master.name}"),
        master=esc(str(master)), built=esc(built.replace("T", " ").replace("Z", " UTC")),
        files=f"{n_files:,}", categories=len(cats),
        toc=" ".join(toc), body="".join(rows), made=esc(datetime.now().strftime("%Y-%m-%d %H:%M")),
        cats=json.dumps([c for c in cats]), key=json.dumps(f"fourier-ratings:{master}"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {"files": n_files, "categories": len(cats), "path": str(out)}


def report_path(master_dir) -> Path:
    from ..paths import fourier_home
    from .why_log import master_key
    return fourier_home() / "reports" / f"report-{Path(master_dir).name}-{master_key(master_dir)[:8]}.html"


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg: #fff; --fg: #1d1d1f; --dim: #6e6e73; --line: #e5e5ea; --accent: #0a66d8;
        --keep: #1a7f37; --drop: #c62828; --mis: #a15c00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161618; --fg: #f2f2f7; --dim: #9a9aa0;
        --line: #2c2c30; --accent: #5aa2ff; --keep: #4cc26a; --drop: #ff6b6b; --mis: #f0a742; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.45 -apple-system, system-ui, sans-serif;
       margin: 0 auto; max-width: 980px; padding: 16px; }}
h1 {{ font-size: 22px; margin: 8px 0 4px; }} h2 {{ font-size: 18px; margin: 28px 0 6px; }}
.n, .meta, .from, .note, .sub {{ color: var(--dim); font-size: 13px; }}
.toc a {{ display: inline-block; margin: 2px 10px 2px 0; color: var(--accent); text-decoration: none; }}
details {{ border-top: 1px solid var(--line); padding: 4px 0; }} summary {{ cursor: pointer; }}
ul {{ list-style: none; margin: 4px 0 8px; padding: 0; }} li {{ padding: 3px 0; overflow-wrap: anywhere; }}
button {{ font: inherit; border: 1px solid var(--line); background: transparent; color: var(--fg);
         border-radius: 6px; padding: 0 7px; cursor: pointer; }}
button.play {{ color: var(--accent); }} .playing .play {{ background: var(--accent); color: var(--bg); }}
.rate {{ float: right; }} .rate button {{ margin-left: 3px; }}
li.keep .rate [data-r=keep] {{ background: var(--keep); color: var(--bg); }}
li.drop .rate [data-r=drop] {{ background: var(--drop); color: var(--bg); }}
li.misfiled .rate [data-r=misfiled] {{ background: var(--mis); color: var(--bg); }}
.bar {{ position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line); z-index: 1; }}
</style></head><body>
<h1>{title}</h1>
<p class="sub">{files} files in {categories} categories, built {built}, at {master}. Page made {made}.</p>
<p>Click &#9654; to listen. <b>K</b>eep, <b>D</b>rop or <b>M</b>isfiled teach the next build:
they're remembered in this browser, and <b>Export ratings</b> saves a CSV for
<code>fourier review import ratings.csv</code>. Each category says what the last build left out.</p>
<div class="bar"><button id="export">Export ratings (<span id="count">0</span>)</button>
<button id="stop">Stop</button> <span class="toc">{toc}</span></div>
{body}
<script>
const KEY = {key}, CATS = {cats};
let ratings = {{}};
try {{ ratings = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(ratings)); }} catch (e) {{}}
  document.getElementById("count").textContent = Object.keys(ratings).length; }};
const show = li => {{ li.classList.remove("keep", "drop", "misfiled");
  const r = ratings[li.dataset.path]; if (r) li.classList.add(r.rating); }};
document.querySelectorAll("li[data-path]").forEach(show); save();
const audio = new Audio(); let playing = null;
audio.onended = () => playing && playing.classList.remove("playing");
document.body.addEventListener("click", ev => {{
  const b = ev.target.closest("button"); if (!b) return;
  const li = b.closest("li");
  if (b.classList.contains("play")) {{
    if (playing) playing.classList.remove("playing");
    audio.src = b.dataset.src; audio.play(); playing = li; li.classList.add("playing");
  }} else if (b.dataset.r) {{
    const p = li.dataset.path, r = b.dataset.r;
    if (ratings[p] && ratings[p].rating === r) delete ratings[p];
    else {{
      let to = "";
      if (r === "misfiled") {{
        to = (prompt("Which category does it belong in? " + CATS.join(", "), "") || "").trim().toUpperCase();
        if (!CATS.includes(to)) return;
      }}
      ratings[p] = {{rating: r, category: to}};
    }}
    show(li); save();
  }} else if (b.id === "stop") {{ audio.pause(); if (playing) playing.classList.remove("playing"); }}
  else if (b.id === "export") {{
    const rows = [["path", "rating", "category"]].concat(
      Object.entries(ratings).map(([p, r]) => [p, r.rating, r.category || ""]));
    const csv = rows.map(r => r.map(x => /[",\\n]/.test(x) ? '"' + x.replace(/"/g, '""') + '"' : x).join(",")).join("\\n") + "\\n";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
    a.download = "ratings.csv"; a.click();
  }}
}});
</script></body></html>
"""
