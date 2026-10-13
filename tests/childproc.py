"""Running a child Python the way the end-to-end tests do (the synthetic build, the CLI runner).

DIAGNOSTIC RETRY, to remove once the flake stays gone: CI on macOS runners failed three times
(2026-10-06 on 3.12, 2026-10-06 on 3.14, 2026-10-12 on 3.14) in such a child, which couldn't
import an installed package ("No module named 'click'", "'click.exceptions'" raised from
click/types.py, setuptools' "_distutils_hack"); a rerun always passed, Linux never failed. CI
now precompiles the venv and runs with PYTHONDONTWRITEBYTECODE (ci.yml), the first suspect. On
such a failure `run_child` runs the same interpreter with the same env and cwd to report what
the child sees (its prefix, sys.path, where the missing module resolves, the package folder the
parent imports it from), warns with that (the pytest summary shows it) and runs the command
once more; failing the same way twice is an AssertionError carrying both outputs and the facts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence

# a traceback's last line, or the CLI's own line for an error no command handled (cli/_app.py)
_MISSING = re.compile(r"(?:ModuleNotFoundError: |stopped: )No module named '([^']+)'")
TAIL = 3000

# what the child's own interpreter sees; argv[1:] are the modules to find
_PROBE = """
import importlib.util, json, os, sys
facts = {"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
         "path": sys.path, "flags.no_user_site": sys.flags.no_user_site,
         "env": {k: v for k, v in os.environ.items() if k.startswith("PYTHON") or k == "VIRTUAL_ENV"},
         "specs": {}}
for name in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(name)
        facts["specs"][name] = spec and {"origin": spec.origin,
                                         "locations": list(spec.submodule_search_locations or [])}
    except Exception as e:
        facts["specs"][name] = repr(e)
print(json.dumps(facts, indent=1, default=str))
"""


class ChildImportFlake(UserWarning):
    """A child Python couldn't import an installed module, and was run once more."""


def missing_modules(r: subprocess.CompletedProcess) -> list[str]:
    """The modules a failed child reported missing, in order; none when it exited 0."""
    if r.returncode == 0:
        return []
    return list(dict.fromkeys(_MISSING.findall((r.stdout or "") + (r.stderr or ""))))


def _package_listing(name: str) -> str:
    """The folder this (the parent) process imports the module's top package from, listed."""
    import importlib.util
    top = name.split(".")[0]
    try:
        spec = importlib.util.find_spec(top)
    except Exception as e:
        return f"{top}: {e!r}"
    if spec is None:
        return f"{top}: not importable in the parent either"
    folder = (list(spec.submodule_search_locations or []) or [os.path.dirname(spec.origin or "")])[0]
    try:
        names = sorted(os.listdir(folder))
    except OSError as e:
        return f"{top} at {folder}: {e!r}"
    return f"{top} at {folder} ({len(names)} entries): {', '.join(names[:80])}"


def child_facts(python: str, missing: Sequence[str], *, env: Mapping[str, str] | None = None,
                cwd: str | os.PathLike | None = None) -> str:
    """What `python` run with this env and cwd sees, and where the parent finds the modules."""
    names = dict.fromkeys([*missing, *(m.split(".")[0] for m in missing)])    # and their packages
    try:
        r = subprocess.run([python, "-c", _PROBE, *names], env=env, cwd=cwd, capture_output=True,
                           text=True, timeout=120, check=False)
        child = r.stdout if r.returncode == 0 else f"(the probe failed: {r.returncode})\n{r.stdout}{r.stderr}"
    except (OSError, subprocess.TimeoutExpired) as e:
        child = f"(the probe didn't run: {e!r})"
    try:
        child = json.dumps(json.loads(child), indent=1)
    except ValueError:
        pass
    parent = "\n".join(_package_listing(m) for m in missing)
    return (f"child's view:\n{child}\nparent: {sys.executable} (prefix {sys.prefix})\n{parent}")


def _out(r: subprocess.CompletedProcess) -> str:
    return f"[exit {r.returncode}]\n{(r.stdout or '')[-TAIL:]}{(r.stderr or '')[-TAIL:]}"


def run_child(cmd: Sequence[str], *, env: Mapping[str, str] | None = None,
              cwd: str | os.PathLike | None = None,
              timeout: float | None = None) -> subprocess.CompletedProcess:
    """subprocess.run(cmd, capture_output=True, text=True), cmd[0] being a Python. A nonzero
    exit that reports a ModuleNotFoundError is diagnosed and retried once (module docstring);
    any other result is returned as it is, for the caller to check."""
    def run() -> subprocess.CompletedProcess:
        return subprocess.run(list(cmd), env=env, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, check=False)

    first = run()
    missing = missing_modules(first)
    if not missing:
        return first
    facts = child_facts(cmd[0], missing, env=env, cwd=cwd)
    warnings.warn(ChildImportFlake(
        f"diagnostic retry (the macOS import flake): {shlex.join(map(str, cmd))} could not import "
        f"{', '.join(missing)}; running it once more\n{facts}\nfirst run: {_out(first)}"), stacklevel=2)
    second = run()
    if missing_modules(second):
        raise AssertionError(f"{shlex.join(map(str, cmd))} could not import {', '.join(missing)}, "
                             f"twice\nfirst run: {_out(first)}\nretry: {_out(second)}\n{facts}")
    return second
