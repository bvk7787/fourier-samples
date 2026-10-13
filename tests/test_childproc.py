"""tests/childproc.py: a child Python that can't import an installed module is diagnosed with its
own facts and run once more (the diagnostic retry for the macOS import flake)."""
import os
import subprocess
import sys
import warnings

import pytest

from childproc import ChildImportFlake, missing_modules, run_child

# fails like the flake (a ModuleNotFoundError, exit 1) while its marker file holds a count > 0
FLAKY = """
import sys
from pathlib import Path
marker = Path(sys.argv[1])
left = int(marker.read_text())
marker.write_text(str(left - 1))
if left > 0:
    print("starting the build")
    raise ModuleNotFoundError("No module named 'qxvwz.exceptions'", name="qxvwz.exceptions")
print("built", left)
"""


def _script(tmp_path, failures):
    (tmp_path / "flaky.py").write_text(FLAKY)
    (tmp_path / "marker").write_text(str(failures))
    return [sys.executable, str(tmp_path / "flaky.py"), str(tmp_path / "marker")]


def test_a_first_run_that_cannot_import_is_diagnosed_and_retried(tmp_path):
    env = {**os.environ, "PYTHONQXV": "set by the test"}
    with pytest.warns(ChildImportFlake) as caught:
        r = run_child(_script(tmp_path, 1), env=env, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "built 0"
    (w,) = caught
    msg = str(w.message)
    assert "diagnostic retry" in msg and "qxvwz.exceptions" in msg
    # the child's own facts: its interpreter, prefix, path, the PYTHON* variables it got, and
    # where the missing module resolves (nowhere)
    assert f'"executable": "{sys.executable}"' in msg and '"base_prefix"' in msg and '"path"' in msg
    assert '"PYTHONQXV": "set by the test"' in msg and '"PYTHONNOUSERSITE": "1"' in msg
    assert '"qxvwz.exceptions": "ModuleNotFoundError' in msg or '"qxvwz.exceptions": null' in msg
    assert "qxvwz: not importable in the parent either" in msg
    assert "starting the build" in msg                         # the first run's output


def test_the_parent_lists_the_package_folder_it_imports_from(tmp_path):
    r = _fake_missing("click.exceptions")
    with pytest.warns(ChildImportFlake) as caught:
        run_child([sys.executable, "-c", r], cwd=tmp_path)
    msg = str(caught[0].message)
    assert "click at " in msg and "exceptions.py" in msg and "types.py" in msg


def _fake_missing(name):
    """A child that reports `name` missing the first time only (a file beside it remembers)."""
    return ("import os, sys\n"
            "if not os.path.exists('ran'):\n"
            "    open('ran', 'w').close()\n"
            f"    raise ModuleNotFoundError(\"No module named '{name}'\")\n")


def test_failing_the_same_way_twice_raises_with_both_outputs(tmp_path):
    with pytest.warns(ChildImportFlake), pytest.raises(AssertionError) as e:
        run_child(_script(tmp_path, 2), cwd=tmp_path)
    msg = str(e.value)
    assert "could not import qxvwz.exceptions, twice" in msg
    assert msg.count("starting the build") == 2 and "first run: [exit 1]" in msg and "retry: [exit 1]" in msg
    assert '"prefix"' in msg                                   # the facts too


def test_other_results_come_back_as_they_are(tmp_path):
    warnings.simplefilter("error", ChildImportFlake)         # (pytest restores the filters)
    ok = run_child([sys.executable, "-c", "print('fine')"])
    assert ok.returncode == 0 and ok.stdout == "fine\n"
    # a failure that isn't a missing module: no retry, no warning; the caller checks it
    (tmp_path / "count").write_text("")
    code = "import sys; open(sys.argv[1], 'a').write('x'); sys.exit('no such library folder')"
    bad = run_child([sys.executable, "-c", code, str(tmp_path / "count")])
    assert bad.returncode == 1 and (tmp_path / "count").read_text() == "x"
    # a missing module reported by a child that still exits 0 is its own business
    fine = run_child([sys.executable, "-c", "print(\"ModuleNotFoundError: No module named 'x'\")"])
    assert fine.returncode == 0 and missing_modules(fine) == []


def test_the_missing_modules_a_failed_child_reports():
    def done(code, out):
        return subprocess.CompletedProcess([], code, out, "")
    tb = "Traceback ...\nModuleNotFoundError: No module named 'click.exceptions'\n"
    assert missing_modules(done(1, tb)) == ["click.exceptions"]
    # the CLI's one line for an exception no command handled
    assert missing_modules(done(1, "fourier build stopped: No module named 'qxv'\n")) == ["qxv"]
    assert missing_modules(done(1, "fourier doctor stopped: no library\n")) == []
