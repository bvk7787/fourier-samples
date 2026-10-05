"""Capture `fourier demo` and a `fourier why` answer as the SVG at the top of the README.

    python scripts/capture_demo.py [--out docs/images/demo.svg]

Runs the demo in a scratch folder (its own home and config, as always), then asks `fourier
why` about one of its drum loops, and writes both as one terminal picture (rich's SVG export)
with the scratch path shown as ~/fourier-demo. Run it again after a change to what the demo
or `why` prints.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 92
WHY = "Break 128 01"


def _run(args, env) -> str:
    out = subprocess.run([sys.executable, "-m", "fourier", *args], env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "images" / "demo.svg"))
    a = ap.parse_args(argv)
    from rich.console import Console
    from rich.text import Text

    with tempfile.TemporaryDirectory(prefix="fourier-capture-") as tmp:
        demo = Path(tmp) / "fourier-demo"
        env = {k: v for k, v in os.environ.items() if not k.startswith("FOURIER_")}
        env.update(HOME=str(Path(tmp) / "home"), COLUMNS=str(WIDTH), FORCE_COLOR="1",
                   TERM="xterm-256color")
        (Path(tmp) / "home").mkdir()
        shown = _run(["demo", "--dir", str(demo)], env)
        shown = shown.split("\nTry next.")[0].rstrip() + "\n"
        env.update(FOURIER_HOME=str(demo / "home"), FOURIER_CONFIG=str(demo / "fourier.toml"))
        why = _run(["why", WHY], env)
        rec = Console(record=True, width=WIDTH, force_terminal=True, color_system="truecolor")
        for cmd, body in (("fourier demo", shown), (f'fourier why "{WHY}"', why)):
            rec.print(Text.assemble(("$ ", "bold green"), (cmd, "bold")))
            rec.print(Text.from_ansi(body.replace(str(demo), "~/fourier-demo").rstrip("\n")))
            rec.print()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        rec.save_svg(a.out, title="fourier")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
