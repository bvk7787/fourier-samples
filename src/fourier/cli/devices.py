"""fourier devices: the device profiles (list, show, and new: a profile of your own)."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import click
from rich.table import Table

from ..devices.loader import DeviceLoader, DeviceProfileError, user_devices_dir, user_devices_dirs
from ._app import _table_cols, console, log, main  # noqa: F401

# ---------------------------------------------------------------------------
# devices commands
# ---------------------------------------------------------------------------

@main.group(short_help="List, show and make device profiles.")
def devices():
    """The device profiles: what each device plays and its limits, each value cited from the
    device's manual or marked a convention or unverified. The package's are in
    config/devices; your own go in ~/.config/fourier/devices (or the folders in
    $FOURIER_DEVICES), where `fourier devices new` writes them. Your device isn't listed?
    Make a profile with `fourier devices new`.

    \b
      fourier devices list
      fourier devices show m8_tracker
      fourier devices new
    """


@devices.command("list", short_help="List the device profiles.")
def devices_list():
    """List the device profiles: format, how samples are loaded, storage, and whether the
    values are checked against the manual."""
    loader = DeviceLoader()
    device_ids = loader.list_devices()
    if not device_ids:
        console.print("[yellow]No device profiles found in config/devices/[/yellow]")
        return

    table = Table(title="Available Devices")
    table.add_column("ID", style="cyan", no_wrap=True,          # the word you type: never cut
                     min_width=max(len(d) for d in device_ids))
    table.add_column("Name", style="green")
    table.add_column("Format")
    table.add_column("Load")
    table.add_column("Storage", justify="right")
    table.add_column("Checked")
    table.add_column("From")

    for did in device_ids:
        try:
            p = loader.load(did)
            table.add_row(did, p.name, f"{p.sample_rate} Hz / {p.bit_depth}-bit / {p.channels}",
                          p.load, f"{p.storage_mb / 1000:g} GB" if p.storage_mb else "-",
                          "manual" if p.verified else "[yellow]unverified[/yellow]",
                          "yours" if p.origin == "user" else "package")
        except Exception as e:
            table.add_row(did, f"[red]Error: {e}[/red]", "-", "-", "-", "-", "-")

    console.print(table)
    for path, why in loader.problems():
        console.print(f"skipped {path}: {why}", style="yellow", markup=False, highlight=False)
    console.print("Your device isn't listed? `fourier devices new` makes a profile for it.",
                  markup=False, highlight=False)


@devices.command("show", short_help="Show a device profile, each value with its source.")
@click.argument("device_id")
def devices_show(device_id):
    """Show a device profile: each value with the manual page it comes from, or its status
    (convention, unverified)."""
    loader = DeviceLoader()
    try:
        profile = loader.load(device_id)
    except (ValueError, OSError) as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise SystemExit(1) from None
    console.print(profile.summary(), markup=False, highlight=False)
    for msg in profile.warnings:
        console.print(f"warning: {msg}", style="yellow", markup=False, highlight=False)


# ---------------------------------------------------------------------------
# devices new: a profile of your own
# ---------------------------------------------------------------------------

# How to draft and cite a profile: the guide in the repository (an installed copy has no docs/).
DEVICE_DOCS_URL = "https://github.com/bvk7787/fourier-samples/blob/main/docs/device-profiles.md"

LOAD_WAYS = {"card": "card-sync", "transfer": "transfer", "folder": "copy"}
LOAD_HELP = ("card: the device reads an SD card or USB drive (fourier sync copies the render "
             "onto it); transfer: its maker's app sends files and never overwrites one; "
             "folder: you copy the render yourself (a DAW, a computer folder)")


def _path_limit_needed(card_dir: str) -> int:
    """The least path limit that leaves the master's names their room (knobs.min_room)
    under this card folder and the longest category folder."""
    from types import SimpleNamespace

    from ..knobs import min_room, path_room
    from ..packs.curate_config import CATEGORY_ORDER
    p = SimpleNamespace(card_dir=card_dir or "", root="", max_path_length=0)
    return min_room() - path_room(p, CATEGORY_ORDER)


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
    return s if s and s[0].isalpha() else f"device_{s}".rstrip("_")


def profile_yaml(a: dict) -> str:
    """The text of a profile `devices new` writes: every value `status: unverified`, with a
    comment block on how to check it against the manual."""
    def fact(v):
        if isinstance(v, list):
            v = "[" + ", ".join(v) + "]"
        elif v is None:
            v = "null"
        elif isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, str):
            v = f'"{v}"' if v.startswith("/") or not v else v
        return f"{{value: {v}, status: unverified}}"

    lines = [
        f"# Made with `fourier devices new` on {date.today().isoformat()}.",
        "# Every value is `status: unverified`: what you answered, not checked against a manual.",
        "# To check one, find it in the device's manual and cite the page: add the manual",
        "# (manual: {title, version, url, sha256}) and a citation (citations: {id: {claim,",
        "# page, quote}}), then change the value's `status: unverified` to `cite: <id>` and",
        "# remove the `status: unverified` line at the top once every value is cited or a",
        f"# `status: convention`. {DEVICE_DOCS_URL} walks through it, and",
        f"# `fourier devices show {a['id']}` prints each value with its source.",
        f"id: {a['id']}",
        f"name: \"{a['name']}\"",
        "status: unverified",
        "description: >",
        f"  {a['name']}: a profile made with `fourier devices new`; its values are unverified.",
        f"load: {a['load']}",
        "sample_refs: path",
        "",
        "paths:",
    ]
    if a["load"] == "card-sync" and a.get("card_dir"):
        lines.append(f"  card_dir: {fact(a['card_dir'])}")
    lines.append(f"  folder_depth: {fact(a['folder_depth'])}")
    lines.append(f"  files_per_folder: {fact(a['files_per_folder'] or None)}"
                 + ("   # no limit" if not a["files_per_folder"] else ""))
    if a.get("max_path"):
        lines.append(f"  max_path_length: {fact(a['max_path'])}")
    if a.get("max_name"):
        lines.append(f"  max_name_length: {fact(a['max_name'])}")
    if a.get("ascii_names"):
        lines.append(f"  ascii_names: {fact(True)}")
    lines += ["", "audio:",
              f"  sample_rate: {fact(a['sample_rate'])}",
              f"  bit_depth: {fact(a['bit_depth'])}",
              f"  channels: {fact(a['channels'])}"
              + ("   # stereo keeps each file's channels; mono folds every file to one"
                 if a["channels"] == "stereo" else ""),
              f"  formats: {fact([a['format']])}"]
    if a.get("max_seconds"):
        lines.append(f"  max_duration_s: {fact(a['max_seconds'])}")
    lines.append(f"  dither: {fact(True)}")
    if a.get("storage_gb"):
        lines += ["", f"storage_mb: {fact(int(round(a['storage_gb'] * 1000)))}"]
    return "\n".join(lines) + "\n"


@devices.command("new", short_help="Make a profile for a device that isn't listed.")
@click.option("--name", help="The device's name, e.g. \"My Sampler\"")
@click.option("--id", "device_id", help="The profile's id (default: from the name, e.g. my_sampler)")
@click.option("--load", "load_way", type=click.Choice(list(LOAD_WAYS)), help=f"How samples get on: {LOAD_HELP}")
@click.option("--sample-rate", type=click.Choice(["44100", "48000"]), help="The rate the device plays")
@click.option("--bit-depth", type=click.Choice(["8", "16", "24"]), help="Bit depth (8 only if the device takes it)")
@click.option("--channels", type=click.Choice(["stereo", "mono"]),
              help="stereo: keep each file's channels; mono: fold every file to one")
@click.option("--format", "fmt", type=click.Choice(["wav", "aiff"]), help="File format the device reads")
@click.option("--card-dir", help="The folder on the card the render goes in (a card device), e.g. /Samples")
@click.option("--folder-depth", type=click.IntRange(1, 2),
              help="2: CATEGORY/family folders; 1: CATEGORY folders, the family in the file name")
@click.option("--files-per-folder", type=click.IntRange(0), help="Most files in one folder (0: no limit)")
@click.option("--max-path", type=click.IntRange(0),
              help="Longest file path the device takes, card folder included (0: no limit)")
@click.option("--max-name", type=click.IntRange(0), help="Longest file or folder name (0: no limit)")
@click.option("--max-seconds", type=click.FloatRange(0), help="Longest file in seconds (0: no limit)")
@click.option("--storage-gb", type=click.FloatRange(0), help="Storage for samples in GB (0: don't say)")
@click.option("--ascii-names/--any-names", default=None,
              help="Plain-ASCII file and folder names (accents dropped), for devices that show only ASCII")
@click.option("--dir", "out_dir", type=click.Path(file_okay=False),
              help="Folder to write it to (default: ~/.config/fourier/devices, or the first of $FOURIER_DEVICES)")
@click.option("--yes", "-y", is_flag=True, help="Ask nothing: the default for every value not given")
@click.option("--force", is_flag=True, help="Replace a profile of your own with the same id")
def devices_new(name, device_id, load_way, sample_rate, bit_depth, channels, fmt, card_dir,
                folder_depth, files_per_folder, max_path, max_name, max_seconds, storage_gb,
                ascii_names, out_dir, yes, force):
    """Make a device profile of your own: asks what the device plays and how its files are
    organized (each answer can be given as an option instead), writes it with every value
    `status: unverified` to ~/.config/fourier/devices/<id>.yaml, and loads it to check it.
    The comment at its top says how to cite the manual (docs/device-profiles.md, on GitHub:
    https://github.com/bvk7787/fourier-samples/blob/main/docs/device-profiles.md).

    \b
      fourier devices new
      fourier devices new --name "My Sampler" --load card --card-dir /Samples \\
          --sample-rate 48000 --bit-depth 16 --max-path 255 --yes
    """
    def stop(msg):
        console.print(msg, style="red", markup=False, highlight=False)
        raise SystemExit(1)

    def ask(given, text, default, type=None):
        if given is not None:
            return given
        if yes:
            return default
        if type is bool:
            return click.confirm(text, default=default)
        return click.prompt(text, default=default, type=type, show_default=True)

    name = (ask(name, "Device name", "My Sampler") or "").strip()
    if not name:
        stop("the device needs a name")
    device_id = ask(device_id, "Profile id", _slug(name))
    if not re.fullmatch(r"[a-z][a-z0-9_]*", device_id or ""):
        stop(f"id {device_id!r}: lowercase letters, digits and _, starting with a letter")
    package = set(DeviceLoader(user_dirs=[]).list_devices())
    if device_id in package:
        stop(f"{device_id} is a package profile; pick another id (or copy it and set "
             f"`override: true`, {DEVICE_DOCS_URL})")
    if not yes and load_way is None:
        console.print(f"How samples get on the device. {LOAD_HELP}.", markup=False, highlight=False)
    load_way = ask(load_way, "How samples load", "card", click.Choice(list(LOAD_WAYS)))
    sample_rate = int(ask(sample_rate, "Sample rate", "44100", click.Choice(["44100", "48000"])))
    bit_depth = int(ask(bit_depth, "Bit depth (8 only if the device takes it)", "16",
                        click.Choice(["8", "16", "24"])))
    channels = ask(channels, "Stereo (keep each file's channels) or mono", "stereo",
                   click.Choice(["stereo", "mono"]))
    fmt = ask(fmt, "File format", "wav", click.Choice(["wav", "aiff"]))
    if load_way == "card":
        card_dir = ask(card_dir, "Folder on the card for the render", "/Samples")
        card_dir = "/" + card_dir.strip().strip("/") if card_dir.strip().strip("/") else ""
    else:
        card_dir = ""
    folder_depth = ask(folder_depth, "Folder levels (2: CATEGORY/family, 1: CATEGORY only)", 2,
                       click.IntRange(1, 2))
    files_per_folder = ask(files_per_folder, "Most files in one folder (0: no limit)", 128, click.IntRange(0))
    max_path = ask(max_path, "Longest file path, card folder included (0: no limit)", 0, click.IntRange(0))
    if max_path:
        need = _path_limit_needed(card_dir)
        if max_path < need:
            stop(f"a path limit of {max_path} leaves too little room for the master's folder and "
                 f"file names under {card_dir or 'the card root'} and its category folders: use at "
                 f"least {need} (--max-path), or 0 for no limit. If the device really allows no "
                 f"more, a shorter card folder (--card-dir) leaves more room.")
    max_name = ask(max_name, "Longest file or folder name (0: no limit)", 0, click.IntRange(0))
    from ..knobs import NAME_ROOM, STEM_MIN
    if max_name and max_name < NAME_ROOM + STEM_MIN:
        stop(f"a name limit of {max_name} leaves too little room for a name (after \".wav\" and a "
             f"\"_2\"): use at least {NAME_ROOM + STEM_MIN} (--max-name), or 0 for no limit")
    max_seconds = ask(max_seconds, "Longest file in seconds (0: no limit)", 0.0, click.FloatRange(0))
    storage_gb = ask(storage_gb, "Storage for samples in GB (0: don't say)", 0.0, click.FloatRange(0))
    ascii_names = ask(ascii_names, "Plain-ASCII names (for a device that shows only ASCII)",
                      load_way == "card", bool)

    folder = Path(out_dir).expanduser() if out_dir else user_devices_dir()
    target = folder / f"{device_id}.yaml"
    mine = set(DeviceLoader(user_dirs=[folder]).list_devices()) - package
    if (target.exists() or device_id in mine) and not force:
        stop(f"{target if target.exists() else device_id} already exists; --force replaces it")
    text = profile_yaml(dict(id=device_id, name=name.replace('"', "'"), load=LOAD_WAYS[load_way],
                             card_dir=card_dir, folder_depth=folder_depth,
                             files_per_folder=files_per_folder, max_path=max_path, max_name=max_name,
                             ascii_names=ascii_names, sample_rate=sample_rate, bit_depth=bit_depth,
                             channels=channels, format=fmt, max_seconds=max_seconds,
                             storage_gb=storage_gb))
    folder.mkdir(parents=True, exist_ok=True)
    old = target.read_text() if target.exists() else None
    target.write_text(text)
    try:
        profile = DeviceLoader(user_dirs=[folder]).load(device_id)
    except (DeviceProfileError, ValueError) as e:
        if old is None:
            target.unlink()
        else:
            target.write_text(old)
        stop(f"the profile didn't load, so it wasn't kept: {e}")
    console.print(profile.summary(), markup=False, highlight=False)
    console.print(f"\nWrote {target}", markup=False, highlight=False)
    if folder.resolve() not in {d.resolve() for d in user_devices_dirs()}:
        console.print(f"Fourier reads your profiles from {', '.join(map(str, user_devices_dirs())) or 'nowhere'} "
                      f"(FOURIER_DEVICES); add {folder} to $FOURIER_DEVICES to use this one.",
                      style="yellow", markup=False, highlight=False)
    console.print(f"Next: `fourier render {device_id}`, and add \"{device_id}\" to devices = [...] in "
                  f"fourier.toml so builds size names for it. To check it against the manual: "
                  f"{DEVICE_DOCS_URL}", markup=False, highlight=False)
