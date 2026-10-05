"""fourier open: show one of Fourier's folders (or the config file) in Finder or the
system's file browser, so nobody has to find a hidden folder by hand."""
from __future__ import annotations

import os

import click

from ._app import console, main

PLACES = ("master", "report", "renders", "releases", "logs", "config", "home")


def place_path(what: str, device: str | None = None) -> str:
    """The folder (or, for config, the file) `fourier open WHAT` shows."""
    from .. import places
    from ..paths import fourier_home
    if what == "master":
        return places.master_dir()
    if what == "renders":
        return places.renders_dir(device)
    if what == "releases":
        return places.releases_root()
    if what == "logs":
        return str(fourier_home() / "logs")
    if what == "home":
        return str(fourier_home())
    raise ValueError(what)


@main.command("open", short_help="Show the master, its report page, a render or the logs.")
@click.argument("what", type=click.Choice(PLACES), default="master", required=False)
@click.argument("device", required=False)
@click.option("--print", "print_only", is_flag=True,
              help="Print the path instead of opening it")
@click.pass_context
def open_cmd(ctx, what, device, print_only):
    """Show one of Fourier's folders in your file browser (Finder on a Mac): the master
    (the default), the renders (one device's with DEVICE), the releases, the build logs, the
    Fourier home, or `config`: your fourier.toml in a text editor (as `fourier config edit`).
    `report` writes a page about the master and opens it in your web browser: every
    category and family with a play button for each file, where it came from, what the last
    build left out and why, and Keep / Drop / Misfiled buttons that export a CSV for
    `fourier review import`.

    \b
      fourier open                      # the master
      fourier open report               # listen, understand and rate in your web browser
      fourier open renders digitakt_2   # the folder to drag into Elektron Transfer
      fourier open logs                 # every build's full log
    """
    from ..platforms import open_path
    if what == "config":
        if print_only:
            from .. import layers
            p = layers.find_config(((ctx.find_root().obj or {}).get("config_args") or (None,))[0])
            click.echo(str(p or layers.USER_CONFIG.expanduser()))
            return
        from .config import config_edit
        ctx.invoke(config_edit)
        return
    from ._app import config_error_stop
    config_error_stop(ctx)                     # the folders come from fourier.toml
    if device and what != "renders":
        raise click.UsageError("DEVICE goes with `renders` only (fourier open renders digitakt_2)")
    if what == "report":
        from ..packs.report import build_report, report_path
        from .releases import need_master
        master = need_master(None)
        r = build_report(master, report_path(master))
        if print_only:
            click.echo(r["path"])
            return
        if open_path(r["path"]) is None:
            console.print(f"Wrote {r['path']}: open it in your web browser.", markup=False,
                          highlight=False, soft_wrap=True)
            return
        console.print(f"Opened the report ({r['files']:,} files): {r['path']}", markup=False,
                      highlight=False, soft_wrap=True)
        return
    path = place_path(what, device)
    if print_only:
        click.echo(path)
        return
    if not os.path.isdir(path):
        nxt = {"master": "`fourier build` makes it",
               "renders": f"`fourier render {device or '<device>'}` makes it",
               "releases": "`fourier publish` makes the first release",
               "logs": "a build on a terminal writes its log there",
               "home": "the first `fourier` command that needs it makes it"}[what]
        console.print(f"Nothing there yet ({path}): {nxt}.", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    if open_path(path) is None:
        console.print(f"No file browser to open it with here: it's {path}.", markup=False,
                      highlight=False, soft_wrap=True)
        raise SystemExit(1)
    console.print(f"Opened {path}", markup=False, highlight=False, soft_wrap=True)
