"""Files gone, new, moved and with changed audio between two master manifests (a stable
rebuild shows none). python scripts/manifest_diff.py <old manifest.json> <new manifest.json>"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourier.packs.manifests import read  # noqa: E402  (sources absolute, any manifest format)

a, b = read(os.path.expanduser(sys.argv[1])), read(os.path.expanduser(sys.argv[2]))
def ents(m):
    out = {}
    for c, v in m["categories"].items():
        for e in v["entries"]:
            out[e["src"]] = (c, e["out"], e.get("out_md5"))
    return out
A, B = ents(a), ents(b)
print("gone", [A[k][:2] for k in A if k not in B][:10])
print("new", [B[k][:2] for k in B if k not in A][:10])
ch = [(B[k][0], B[k][1]) for k in B if k in A and A[k][2] != B[k][2]]
print("audio changed", len(ch), ch[:15])
mv = [(A[k][1], B[k][1]) for k in B if k in A and A[k][1] != B[k][1] and A[k][2] == B[k][2]]
print("moved/renamed", len(mv), mv[:5])
for s in ("SLICE", "KITS"):
    sa = {e["out"]: e.get("out_md5") for e in a["sets"][s]["entries"]}; sb = {e["out"]: e.get("out_md5") for e in b["sets"][s]["entries"]}
    print(s, "added", len(set(sb) - set(sa)), "removed", len(set(sa) - set(sb)), "changed", sum(1 for k in sb if k in sa and sa[k] != sb[k]))
