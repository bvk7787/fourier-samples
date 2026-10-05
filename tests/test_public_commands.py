"""The command surface: what `fourier --help` lists and in which sections, the removed names
staying gone, and the commands that stand for several steps (build, verify, render,
publish, tools analyze, tools scan)."""
import json

import click
import pytest
from click.testing import CliRunner

from fourier.cli import _app, main
from fourier.cli import build as B
from fourier.cli import enrich as E
from fourier.cli import ingest as I
from fourier.cli import releases as R

TOP = {"setup", "demo", "doctor", "build", "render", "sync", "publish", "releases", "open", "why", "verify",
       "diff", "search", "config", "devices", "review", "tools"}
TOOLS = {"scan", "analyze", "train", "audit", "dedup", "import-folder", "resolve", "db-stats"}
REMOVED = ("init", "plan", "scan", "analyze", "pack", "import", "enrich", "db", "library",
           "categories", "play")


def _walk(cmd, path="fourier"):
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _walk(sub, f"{path} {name}")


def test_the_top_level_and_tools():
    assert set(main.commands) == TOP
    assert set(main.commands["tools"].commands) == TOOLS
    assert [p for p, c in _walk(main) if c.hidden] == []           # no hidden aliases


def test_help_lists_the_commands_in_sections():
    out = CliRunner().invoke(main, ["--help"], terminal_width=100).output
    titles = [t for t, _names in _app.SECTIONS]
    assert titles == ["Get started", "Build and load", "Releases", "Look inside", "Settings",
                      "Listen and rate", "Advanced"]
    at = [out.index(f"\n{t}:\n") for t in titles]
    assert at == sorted(at) and "Commands:" not in out and "Other:" not in out
    for (title, names), start, end in zip(_app.SECTIONS, at, at[1:] + [len(out)]):
        block = out[start:end]
        assert [ln.split()[0] for ln in block.strip().splitlines()[1:] if ln.strip()] == list(names), title
    assert {n for _t, names in _app.SECTIONS for n in names} == TOP
    # one column for every section
    cols = {ln.index(ln.split()[1]) for ln in out.split("Get started:")[1].splitlines()
            if ln.startswith("  ") and len(ln.split()) > 1}
    assert len(cols) == 1


@pytest.mark.parametrize("name", REMOVED)
def test_the_removed_names_are_gone(tmp_path, name):
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), name, "--help"])
    assert r.exit_code == 2 and "No such command" in r.output


def test_every_command_has_a_whole_line_in_the_help():
    """Each help listing shows a command's short help: a whole sentence, not one click cut
    at an abbreviation's period ("(e.g.") or at its width ("...")."""
    for path, cmd in _walk(main):
        if path == "fourier":
            continue
        line = cmd.get_short_help_str(limit=200)
        assert line.endswith(".") and "..." not in line and not line.endswith("e.g."), (path, line)
        assert "..." not in cmd.get_short_help_str(limit=80), path


def test_version():
    from importlib.metadata import version
    r = CliRunner().invoke(main, ["--version"])
    assert r.exit_code == 0 and r.output.strip() == f"fourier (Fourier Samples) {version('fourier-samples')}"


def test_help_examples_keep_their_lines():
    r = CliRunner().invoke(main, ["render", "--help"], terminal_width=80)
    assert "\n    fourier render m8_tracker --check" in r.output


def test_search_takes_a_free_text_query(monkeypatch, tmp_path):
    db = ["--db", str(tmp_path / "t.duckdb")]
    monkeypatch.setenv("FOURIER_CLAP_INDEX", str(tmp_path / "no-index.npz"))
    r = CliRunner().invoke(main, [*db, "search", "dark", "kick"])
    assert r.exit_code == 1 and "needs the CLAP index" in r.output, r.output
    r = CliRunner().invoke(main, [*db, "search", "kick", "--like", "snare"])
    assert r.exit_code == 2 and "not both" in r.output


def _record(monkeypatch, module, name, result=None):
    """Replace module.name with a function that records its arguments."""
    calls = []

    def fake(*a, **kw):
        calls.append((a, kw))
        return result
    monkeypatch.setattr(module, name, fake)
    return calls


def test_verify_quick_runs_the_invariants(monkeypatch, tmp_path):
    val = _record(monkeypatch, B, "_validate")
    ver = _record(monkeypatch, B, "_run_verify", result=True)
    monkeypatch.setattr(B, "_need_master", lambda d: None)
    db = ["--db", str(tmp_path / "t.duckdb")]
    CliRunner().invoke(main, [*db, "verify", "--quick", "--no-db"], catch_exceptions=False)
    CliRunner().invoke(main, [*db, "verify", "--render", "m8_tracker"], catch_exceptions=False)
    assert val == [((None, True), {})]
    assert ver[0][0][1] == ("m8_tracker",) and len(ver) == 1


def test_render_check_dry_run_and_release(monkeypatch, tmp_path):
    from fourier.packs import render as PR
    chk = _record(monkeypatch, R, "_check_paths")
    ren = _record(monkeypatch, PR, "render_device", result={})
    plan = _record(monkeypatch, PR, "device_plan", result={"safe": False})
    monkeypatch.setattr(R, "need_master", lambda d: d or str(tmp_path / "master"))
    db = ["--db", str(tmp_path / "t.duckdb")]
    CliRunner().invoke(main, [*db, "render", "m8_tracker", "--check"], catch_exceptions=False)
    r = CliRunner().invoke(main, [*db, "render", "m8_tracker", "--dry-run"], catch_exceptions=False)
    assert r.exit_code == 1                       # a path on the device would lose its audio
    CliRunner().invoke(main, [*db, "render", "m8_tracker", "--release", "v1"], catch_exceptions=False)
    assert chk[0][0][0] == "m8_tracker" and len(chk) == 1
    assert plan[0][0][1].device_id == "m8_tracker" and len(plan) == 1
    assert ren[0][1]["release"] == "v1" and len(ren) == 1


def _master(root):
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"fourier_manifest": 2, "categories": {}}))
    return root


def test_publish_dry_run_previews_against_the_latest_release(monkeypatch, tmp_path):
    from fourier.packs import releases as PRel
    master = _master(tmp_path / "master")
    pub = tmp_path / "pub"
    args = ["--db", str(tmp_path / "t.duckdb"), "publish", "--from", str(master), "--to", str(pub),
            "--no-verify", "--dry-run"]
    plan = _record(monkeypatch, PRel, "plan", result={"safe": True})
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 0 and "No release yet" in r.output and plan == [], r.output
    (pub / "releases" / "v3").mkdir(parents=True)
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 0 and plan[0][0][1:] == ("v3", str(master)), r.output
    plan.clear()
    monkeypatch.setattr(PRel, "plan", lambda s, base, m, log: plan.append(base) or {"safe": False})
    r = CliRunner().invoke(main, args + ["--base", "v2"])
    assert r.exit_code == 1 and plan == ["v2"]
    assert not (pub / "releases" / "v4").exists()


def test_tools_analyze_runs_the_steps_in_order(monkeypatch, tmp_path):
    ran = []
    for step in E.ANALYZE_STEPS:
        monkeypatch.setattr(E.steps.commands[step], "callback",
                            lambda _s=step, **kw: ran.append((_s, kw)))
    db = ["--db", str(tmp_path / "t.duckdb")]
    CliRunner().invoke(main, [*db, "tools", "analyze", "--workers", "8"], catch_exceptions=False)
    assert [s for s, _ in ran] == list(E.ANALYZE_STEPS)
    kw = dict(ran)
    assert kw["librosa"]["workers"] == 8 and kw["clap"]["build_index_flag"] is True
    assert "workers" not in kw["derived"]
    ran.clear()
    CliRunner().invoke(main, [*db, "tools", "analyze", "--only", "key", "--only", "derived"],
                       catch_exceptions=False)
    assert [s for s, _ in ran] == ["derived", "key"]


def test_tools_scan_walks_the_library_without_sononym_or_live(monkeypatch, tmp_path):
    lib = tmp_path / "Samples"
    lib.mkdir()
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    from fourier import places
    from fourier.ingest import ableton_tags
    places.reset()
    monkeypatch.setattr(ableton_tags, "latest_live_db", lambda: None)
    son = _record(monkeypatch, I, "_sononym_sync")
    walk = _record(monkeypatch, I, "_walk")
    CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"], catch_exceptions=False)
    places.reset()
    assert son == [] and walk[0][0][0] == lib


def test_tools_scan_with_a_library_named_by_name_only(monkeypatch, tmp_path):
    from fourier import places
    from fourier.ingest import ableton_tags
    monkeypatch.setenv("FOURIER_LIBRARY", "SampleLibrary")
    places.reset()
    monkeypatch.setattr(ableton_tags, "latest_live_db", lambda: None)
    walk = _record(monkeypatch, I, "_walk")
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"])
    places.reset()
    assert r.exit_code == 0 and walk == [] and "by name only" in r.output, r.output


def test_build_refuses_without_a_library(monkeypatch, tmp_path):
    # without `library` the build would read packs from the database's paths and verify from
    # the master's absolute paths, and every pack rule would disagree
    from fourier import places
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    monkeypatch.setenv("FOURIER_CONFIG", "none")
    places.reset()
    try:
        r = CliRunner().invoke(main, ["build", "KICKS", "--out", str(tmp_path / "m")])
    finally:
        places.reset()
    assert r.exit_code == 2, r.output
    assert "No sample folder set up yet" in r.output


@pytest.fixture
def stages(monkeypatch, tmp_path):
    """A build's scan and analysis recorded instead of run; the CLAP model there."""
    from fourier import places
    lib = tmp_path / "Samples"
    (lib / "Acme").mkdir(parents=True)          # a build stops on an empty library folder
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    places.reset()
    ran = []
    monkeypatch.setattr(I, "run_scan", lambda *a, **k: ran.append("scan"))
    monkeypatch.setattr(E, "run_analysis", lambda *a, **k: ran.append("analyze"))
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)
    yield ran
    places.reset()


def _build(tmp_path, *args):
    return CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "build", *args,
                                     "--out", str(tmp_path / "m")])


@pytest.mark.parametrize("args", [["--all"], ["KICKS"], ["--base", "v1"]])
def test_build_scans_and_analyzes_first(stages, tmp_path, args):
    r = _build(tmp_path, *args)
    assert stages == ["scan", "analyze"], r.output
    out = " ".join(r.output.split())
    assert "1/4 Scan the library" in out and "2/4 Analyze what's new" in out and "3/4 Build" in out


@pytest.mark.parametrize("args", [["--all"], ["KICKS"], ["--base", "v1"]])
def test_build_no_scan_builds_the_database_as_it_is(stages, tmp_path, args):
    r = _build(tmp_path, *args, "--no-scan")
    assert stages == [] and "1/2  Build" in r.output, r.output


def test_build_dry_run_reads_only(stages, tmp_path):
    r = _build(tmp_path, "--all", "--dry-run")
    assert r.exit_code == 0 and stages == [], r.output
    assert not (tmp_path / "t.duckdb").exists() and not (tmp_path / "m").exists()


def test_build_without_the_clap_model_stops_before_scanning(stages, monkeypatch, tmp_path):
    monkeypatch.setattr(E, "clap_extra_missing", lambda: True)
    r = _build(tmp_path, "--all")
    assert r.exit_code == 1 and stages == [], r.output
    assert "run `fourier setup`" in " ".join(r.output.split())


def test_build_estimates_a_big_analysis(stages, monkeypatch, tmp_path):
    from fourier.cli import setup as S
    monkeypatch.setattr(S, "pending_analysis", lambda session=None: {"derived": 1200, "librosa": 1200,
                                                                      "clap": 1200})
    r = _build(tmp_path, "--all")
    assert "1,200 new samples to analyze, " in " ".join(r.output.split()), r.output
