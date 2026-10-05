"""The command tree: every command, its help text and its options, pinned by a snapshot.

A change to the CLI's surface shows up here as a diff of tests/data/cli_tree.json. When the
change is on purpose, regenerate it:  python tests/test_cli_tree.py --write
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import click

SNAPSHOT = Path(__file__).parent / "data" / "cli_tree.json"


def _text(s):
    """Help text without its docstring indent (click versions differ in stripping it)."""
    return inspect.cleandoc(s) if s else s


def _param(p: click.Parameter) -> dict:
    d = {"name": p.name, "kind": type(p).__name__, "opts": list(p.opts) + list(p.secondary_opts),
         "type": getattr(p.type, "name", type(p.type).__name__), "required": p.required,
         "multiple": p.multiple, "nargs": p.nargs}
    default = p.default
    d["default"] = None if callable(default) else repr(default)
    if d["default"] == "Sentinel.UNSET":    # click 8.5 leaves an unset default unset
        d["default"] = "False" if getattr(p, "is_flag", False) and not p.multiple else "None"
    if isinstance(p, click.Option):
        d.update(is_flag=p.is_flag, hidden=p.hidden, help=_text(p.help),
                 show_default=bool(p.show_default))
        if isinstance(p.type, click.Choice):
            d["choices"] = list(p.type.choices)
    return d


def tree(cmd: click.Command | None = None, path: str = "fourier") -> dict:
    if cmd is None:
        from fourier.cli import main as cmd
    out = {path: {"group": isinstance(cmd, click.Group), "hidden": cmd.hidden, "help": _text(cmd.help),
                  "short_help": _text(cmd.short_help), "params": [_param(p) for p in cmd.params]}}
    if isinstance(cmd, click.Group):
        for name in sorted(cmd.commands):
            out.update(tree(cmd.commands[name], f"{path} {name}"))
    return out


def test_cli_tree_matches_snapshot():
    got = tree()
    want = json.loads(SNAPSHOT.read_text())
    assert sorted(got) == sorted(want), "commands added or removed"
    for path in want:
        assert got[path] == want[path], f"{path} changed"


if __name__ == "__main__":
    if "--write" in sys.argv:
        SNAPSHOT.write_text(json.dumps(tree(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {SNAPSHOT}")
