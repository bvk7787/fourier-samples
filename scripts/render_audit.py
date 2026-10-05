"""Independent checks of a device render (a second opinion next to `fourier verify --render`):
hard starts, loop-point jumps, long quiet tails, peaks, tiny cycles, unpaired brackets, flipped
stereo, bars at the folder tempo. python scripts/render_audit.py <render dir> <master dir>"""
import collections
import json
import os
import sys

import numpy as np
import soundfile as sf

render, master = [os.path.expanduser(p) for p in sys.argv[1:3]]
m = json.load(open(os.path.join(master, "manifest.json")))
res = collections.defaultdict(list)
files = []
for d, _, fs in os.walk(render):
    for f in fs:
        if f.endswith(".wav"):
            files.append(os.path.join(d, f))
for p in files:
    rel = os.path.relpath(p, render)
    top = rel.split("/")[0]
    name = os.path.basename(p)
    if name.count("[") != name.count("]") or name.count("(") != name.count(")"):
        res["unpaired brackets"].append(rel)
    y, sr = sf.read(p, always_2d=True)
    if not y.size:
        res["empty"].append(rel)
        continue
    a = np.abs(y).max(axis=1)
    pk = float(a.max())
    if 20 * np.log10(pk + 1e-12) > -0.5:
        res["peak over -0.5 dBFS"].append(rel)
    waves, loop = "WAVES" in top, ("DRUMLOOPS" in top or "PHRASES" in top or "SLICE" in top)
    if waves:
        if len(y) < 256:
            res["cycles under 256"].append((rel, len(y)))
        continue
    if y.shape[1] == 2:
        l, r = y[:, 0], y[:, 1]
        den = float(np.sqrt(np.sum(l * l) * np.sum(r * r)))
        if den > 0 and float(np.sum(l * r)) / den < -0.8:
            res["stereo L/R correlation under -0.8"].append(rel)
    if a[0] > 0.05:
        res["hard start (first sample > 5% FS)"].append((rel, round(float(a[0]), 3)))
    if loop and float(np.abs(y[-1] - y[0]).max()) > 0.2:
        res["loop-point jump > 0.2 FS"].append(rel)
    if not loop:
        q = a < 10 ** (-50 / 20)
        run = len(q) - (np.where(~q)[0][-1] + 1) if (~q).any() else len(q)
        if run / sr > 1.0:
            res["over 1 s under -50 dBFS at the end"].append((rel, round(run / sr, 2)))
# bars at the folder tempo
bad_bars = []
for e in m["categories"]["DRUMLOOPS"]["entries"]:
    t = e.get("bpm_fold") or e.get("bpm")
    if not t or e.get("bars") is None:
        continue
    y_info = sf.info(os.path.join(master, "DRUMLOOPS", e["out"]))
    exact = y_info.frames / y_info.samplerate * t / 240.0
    if abs(exact - e["bars"]) > 0.26:
        bad_bars.append((e["out"], e["bars"], round(exact, 2)))
res["bars not at the folder tempo"] = bad_bars
dl = {"DRUMLOOPS/" + e["out"]: e for e in m["categories"]["DRUMLOOPS"]["entries"]}
sl = m["sets"]["SLICE"]["entries"]     # set entries carry no bars: read their DRUMLOOPS entry
res["SLICE not 1/2/4 bars"] = [e["out"] for e in sl
                               if (dl.get(e.get("came_from") or "") or {}).get("bars") not in (1, 2, 4)]
print(f"{render}: {len(files)} files")
for k in sorted(res):
    v = res[k]
    print(f"  {k}: {len(v)}" + (f"  e.g. {v[:3]}" if v else ""))
