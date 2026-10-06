"""The root command group, config resolution, and what every command module shares."""
from __future__ import annotations

import logging

import click
from rich.console import Console

console = Console()
log = logging.getLogger("fourier")


def _table_cols(table_name: str) -> set[str]:
    """Return column names for a table — cross-DB, no PRAGMA."""
    from ..db.session import get_engine, table_columns
    return table_columns(get_engine(), table_name) or set()


def _resolve_config(ctx, config_path, sets):
    """Resolve the config layers once, before any curation module is imported, and hand
    them to this process and its workers as $FOURIER_RESOLVED_CONFIG (fourier/layers.py).
    A resolved config already in the environment (the golden harness, a parent process)
    wins. Any config error stops here, not mid-build."""
    import os as _os
    from ..settings import ENV, ConfigError, load_all_and_check
    try:
        if config_path:
            # the keys that aren't tunables (providers, preparers) are read wherever they're
            # needed, workers included, from the fourier.toml find_config() names
            from .. import layers
            _os.environ[layers.ENV_CONFIG] = str(layers.find_config(config_path).resolve())
        if _os.environ.get(ENV):
            if config_path or sets:
                console.print(f"[yellow]${ENV} is set, so --config and --set are ignored[/yellow]")
        else:
            from .. import layers
            if config_path or sets or layers.find_config() is not None:
                r = layers.resolve(config_path, sets)
                ctx.obj["resolved"] = r
                if r.values:
                    if ctx.invoked_subcommand == "config":
                        # config only reads: its modules still see the values (config explain
                        # categories lists an overlay's added folders), but from a throwaway
                        # file, so it leaves nothing under $FOURIER_HOME/run
                        import shutil
                        import tempfile
                        from pathlib import Path
                        tmp = tempfile.mkdtemp(prefix="fourier-config-")
                        ctx.call_on_close(lambda: shutil.rmtree(tmp, ignore_errors=True))
                        _os.environ[ENV] = str(layers.write(r, Path(tmp)))
                    else:
                        from ..paths import home_path
                        run = home_path("run")
                        made = [d for d in (run.parent, run) if not d.exists()]
                        new = not (run / f"resolved-{r.hash}.json").exists()
                        _os.environ[ENV] = str(layers.write(r))
                        # a command that turns out to only read (build --dry-run) takes it back
                        ctx.obj["resolved_written"] = (_os.environ[ENV], new, made)
        if _os.environ.get(ENV):
            load_all_and_check()
    except ConfigError as e:
        if ctx.invoked_subcommand in CONFIG_ERROR_OK:
            ctx.obj["config_error"] = str(e)
            return
        say_config_error(str(e))
        raise SystemExit(2) from None


def say_config_error(err: str) -> None:
    console.print(f"config: {err}", style="red", markup=False, highlight=False, soft_wrap=True)
    if ".toml:" in err:
        console.print("`fourier config edit` opens it to fix.", markup=False, highlight=False)


def config_error_stop(ctx) -> None:
    """For a command that tolerates a broken config (CONFIG_ERROR_OK) but needs it here."""
    err = (ctx.find_root().obj or {}).get("config_error")
    if err:
        say_config_error(err)
        raise SystemExit(2)


def read_only_config(ctx=None) -> None:
    """For a command that writes nothing (build --dry-run, doctor, why, diff, render --check
    and --dry-run, publish --dry-run, tools analyze --status): the resolved config this run
    wrote under $FOURIER_HOME/run moves to a temporary folder removed on exit, and the
    folders it made for it go, as `fourier config` has it. Called first thing, so an early
    exit leaves nothing either."""
    import os as _os
    import shutil
    import tempfile
    from pathlib import Path
    from ..settings import ENV
    ctx = ctx or click.get_current_context(silent=True)
    if ctx is None:
        return
    root = ctx.find_root()
    path, new, made = (root.obj or {}).get("resolved_written") or (None, False, [])
    if not path or not new:
        return
    tmp = tempfile.mkdtemp(prefix="fourier-config-")
    moved = Path(tmp) / Path(path).name

    def done():
        shutil.rmtree(tmp, ignore_errors=True)
        if _os.environ.get(ENV) == str(moved):     # a later command in this process resolves anew
            _os.environ.pop(ENV, None)
    root.call_on_close(done)
    shutil.move(path, moved)
    _os.environ[ENV] = str(moved)
    for d in reversed(made):          # the run folder, then the home, when this run made them
        try:
            Path(d).rmdir()
        except OSError:
            pass
    root.obj["resolved_written"] = None


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

HELP_WIDTH = 100


def _help_width() -> int | None:
    """Help text wraps at the terminal's width (at most HELP_WIDTH); piped or redirected,
    there is no terminal, so at HELP_WIDTH rather than click's 80-column fallback."""
    import sys
    try:
        return None if sys.stdout.isatty() else HELP_WIDTH
    except (AttributeError, ValueError):     # a replaced or closed stdout
        return HELP_WIDTH


# commands that set up their own home and config: the root doesn't resolve the user's
# config or open the user's database for them
SELF_CONTAINED = ("demo",)
# commands that write the config: nothing to resolve yet (setup resolves it once written)
WRITES_CONFIG = ("setup",)
# commands that still run with a config that doesn't load (each says so where it matters):
# `config edit` is how it gets fixed, `open config` shows it
CONFIG_ERROR_OK = ("config", "open", "doctor")

# `fourier --help` lists the commands in these sections, in this order
SECTIONS = (
    ("Get started", ("setup", "demo", "doctor")),
    ("Build and load", ("build", "render", "sync")),
    ("Releases", ("publish", "releases")),
    ("Look inside", ("open", "why", "verify", "diff", "search")),
    ("Settings", ("config", "devices")),
    ("Listen and rate", ("review",)),
    ("Advanced", ("tools",)),
)


def unhandled(ctx, e: Exception) -> None:
    """One plain line for an error no command handled, the traceback kept in a file for a
    bug report."""
    import traceback
    from datetime import datetime
    cmd = " ".join(["fourier", *([ctx.invoked_subcommand] if ctx.invoked_subcommand else [])])
    for frame, _line in traceback.walk_tb(e.__traceback__):     # the command that was running
        c = frame.f_locals.get("ctx")
        if isinstance(c, click.Context) and c.info_name:
            cmd = " ".join(["fourier", *c.command_path.split()[1:]])
    if isinstance(e, FileNotFoundError) and e.filename:
        what = f"{e.filename} doesn't exist"
    else:
        what = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    saved = None
    try:
        from ..paths import fourier_home
        d = fourier_home() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        saved = d / f"error-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        saved.write_text("".join(traceback.format_exception(e)))
    except Exception:                            # noqa: BLE001
        saved = None
    console.print(f"{cmd} stopped: {what}", style="red", markup=False, highlight=False,
                  soft_wrap=True)
    console.print("`fourier doctor` checks what's missing." + (
        f" The details, for a bug report: {saved}" if saved else ""), markup=False,
        highlight=False, soft_wrap=True)


class SectionedGroup(click.Group):
    """The root group: `fourier --help` lists its commands under SECTIONS' titles, in that
    order (a command no section names goes under "Other"), with one column width for all.
    An error no command handled ends in one plain line and where the details went, not a
    Python traceback (`fourier -v` or FOURIER_TRACEBACK=1 shows it)."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (click.exceptions.ClickException, click.exceptions.Abort, click.exceptions.Exit,
                SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:                     # noqa: BLE001 (the last stop before a traceback)
            import os as _os
            if ctx.params.get("verbose") or _os.environ.get("FOURIER_TRACEBACK"):
                raise
            unhandled(ctx, e)
            raise SystemExit(1) from None

    def list_commands(self, ctx):
        named = [n for _t, names in SECTIONS for n in names if n in self.commands]
        return named + sorted(set(self.commands) - set(named))

    def format_commands(self, ctx, formatter):
        listed = {n: c for n in self.list_commands(ctx)
                  if (c := self.get_command(ctx, n)) is not None and not c.hidden}
        if not listed:
            return
        width = max(map(len, listed))
        limit = formatter.width - 6 - width
        sections = [(title, [n for n in names if n in listed]) for title, names in SECTIONS]
        named = {n for _t, names in sections for n in names}
        sections.append(("Other", sorted(set(listed) - named)))
        for title, names in sections:
            if not names:
                continue
            with formatter.section(title):
                formatter.write_dl([(n.ljust(width), listed[n].get_short_help_str(limit))
                                    for n in names])


@click.group(cls=SectionedGroup,
             context_settings={"max_content_width": HELP_WIDTH, "terminal_width": _help_width()})
@click.version_option(package_name="fourier-samples", prog_name="fourier",
                      message="%(prog)s (Fourier Samples) %(version)s")
@click.option("--db", default=None,
              help="Path to the library database (default: $FOURIER_HOME/library.duckdb, "
                   "or an older home's library.db)")
@click.option("--config", "config_path", default=None,
              help="fourier.toml to use (default: $FOURIER_CONFIG, ./fourier.toml, "
                   "~/.config/fourier/fourier.toml; FOURIER_CONFIG=none for none)")
@click.option("--set", "sets", multiple=True, metavar="KEY=VALUE",
              help="Override one tunable for this run, over every config layer (repeatable)")
@click.option("--verbose", "-v", is_flag=True,
              help="Debug logging, and the audio libraries' warnings (librosa, numba)")
@click.pass_context
def main(ctx, db, config_path, sets, verbose):
    """Fourier Samples: curate a sample library into device-ready folders for hardware samplers.

    New here? `fourier demo` tries it on a generated library, and `fourier setup` sets it up
    for yours. Each command's --help says more.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if verbose:              # librosa's and numba's warnings too, here and in every worker
        import os as _os
        from .. import VERBOSE_ENV, quiet_library_warnings
        _os.environ[VERBOSE_ENV] = "1"
        quiet_library_warnings(False)
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand in SELF_CONTAINED:
        if db or config_path or sets:
            console.print(f"[yellow]fourier {ctx.invoked_subcommand} uses its own database and "
                          f"config: --db, --config and --set are ignored[/yellow]")
        return
    if ctx.invoked_subcommand not in WRITES_CONFIG:
        _resolve_config(ctx, config_path, sets)
    ctx.obj["config_args"] = (config_path, sets)
    from ..db.session import db_exists, use_db
    use_db(db)                # opened on first use: --help and doctor leave no database behind
    try:
        import os as _os
        from fourier.paths import clap_index_path as _cip
        _idx = str(_cip())
        if _os.path.exists(_idx) and db_exists():
            from fourier.analysis.clap_features import load_index_fast as _lif
            _idx_n = len(_lif(ids_only=True)[0])
            from ..db.session import peek           # read-only: a dry run changes nothing
            _db_n = peek("SELECT COUNT(*) FROM sample_features") or 0
            if _db_n > _idx_n:
                console.print(
                    f"[yellow]Note: {_db_n - _idx_n:,} sample(s) aren't in the CLAP index yet "
                    f"({_idx_n:,} indexed / {_db_n:,} in DB). `fourier build` "
                    f"refreshes it.[/yellow]")
    except Exception:
        pass


@main.group("tools", short_help="Power tools: scan, analyze, train, audit, imports and the database.")
def tools():
    """Power tools for the steps `fourier build` runs for you, and for looking after the
    library and its database.

    \b
      fourier tools scan               # bring new and changed files into the database
      fourier tools analyze --status   # how far each analysis step has got
      fourier tools train              # train the sound model on your library
      fourier tools audit              # what the master leaves out of the library
      fourier tools import-folder ~/Downloads/new-pack
    """
