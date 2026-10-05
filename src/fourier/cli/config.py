"""fourier config: what the config layers resolved to."""
from __future__ import annotations


import click

from ..db.session import session_scope
from ._app import _table_cols, console, log, main  # noqa: F401

# ---------------------------------------------------------------------------
# Config: what the layers resolved to (fourier/layers.py, fourier/settings.py)
# ---------------------------------------------------------------------------

def _tunable_meta():
    """"<module>.<NAME>" -> (class, knob) from config/tunables.yaml."""
    import yaml
    from ..paths import config_dir
    doc = yaml.safe_load((config_dir() / "tunables.yaml").read_text()) or {}
    out = {}
    for m, names in doc.items():
        for n, spec in (names or {}).items():
            cls, _, knob = str(spec).partition(":")
            out[f"{m}.{n}"] = (cls, knob)
    return out


def _plain(v) -> str:
    """A value as it would be written in fourier.toml or read in the code: a regex as its
    pattern, a dict as {key: value}, a list, tuple or set as [a, b]."""
    if getattr(v, "pattern", None) is not None:
        return str(v.pattern)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_plain(k)}: {_plain(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (set, frozenset)):
        return "[" + ", ".join(sorted(_plain(x) for x in v)) + "]"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_plain(x) for x in v) + "]"
    return str(v)


def _short(v, n=70):
    s = _plain(v)
    return s if len(s) <= n else s[:n - 3] + "..."


def _category_examples(n=3) -> dict:
    """{category: [file names]} from the current master's manifest, else from the library
    (files with one of the category's labels), else nothing."""
    import json as _json
    import os as _os
    from ..packs.curate_config import CATEGORIES
    try:
        from ..packs.ratings import live_master_dir
        man = _os.path.join(live_master_dir(), "manifest.json")
        if _os.path.exists(man):
            doc = _json.loads(open(man).read())
            return {c: [_os.path.basename(e.get("out") or "") for e in (v.get("entries") or [])[:n]]
                    for c, v in (doc.get("categories") or {}).items()}
    except Exception:
        pass
    out = {}
    try:
        from sqlalchemy import text as _text
        from ..metadata.rows import fetch, like_any, sample_select
        with session_scope() as s:
            for c, cfg in CATEGORIES.items():
                if not cfg.get("labels"):
                    continue
                sql, p = like_any("canonical", cfg["labels"], "x", quoted=True, session=s)
                q = sample_select("filename", session=s).where(_text(sql).bindparams(**p)).limit(n)
                out[c] = [r.filename for r in fetch(s, q)]
    except Exception:
        pass
    return out


def _explain_categories():
    from .. import taxonomy
    from ..packs.curate_config import CATEGORIES, CATEGORY_ORDER, DERIVED_DIRS, taxonomy_view
    ex = _category_examples()
    console.print(f"[bold]taxonomy[/bold] {taxonomy.TAXONOMY_PATH}")
    for name, d in DERIVED_DIRS.items():
        console.print(f"  {d}  derived set ({name.lower()})")
    doc = taxonomy_view.doc["categories"]     # with a library overlay's added categories
    for i, c in enumerate(CATEGORY_ORDER, 1):
        cfg, t = CATEGORIES[c], doc.get(c, {})
        labels = ", ".join(cfg.get("labels") or ()) or "chosen by other rules"
        roles = ", ".join(t.get("roles") or ())
        console.print(f"  {i:02d}_{c}  {cfg['kind']}; labels: {labels}"
                      + (f"; roles: {roles}" if roles else "")
                      + f"; {len(cfg.get('phrases') or ())} prompts"
                      + (" (added by the config)" if c in taxonomy_view.added else ""))
        if ex.get(c):
            console.print(f"      e.g. {', '.join(ex[c])}")


@main.group("config", short_help="Show, explain or edit your settings (fourier.toml).")
@click.pass_context
def config_cmd(ctx):
    """Your settings: what the config layers resolve to (preset, overlay, fourier.toml,
    --set), what each setting does, and `config edit` to change them.

    \b
      fourier config edit              # open fourier.toml in your text editor
      fourier config show              # what's in effect, and where each value came from
      fourier config explain words     # what one setting does
    """
    if ctx.invoked_subcommand != "edit":         # edit is how a broken config gets fixed
        from ._app import config_error_stop
        config_error_stop(ctx)


@config_cmd.command("edit", short_help="Open your fourier.toml in a text editor.")
@click.pass_context
def config_edit(ctx):
    """Open the fourier.toml in use in a text editor: $VISUAL or $EDITOR when set, else the
    system's (TextEdit on a Mac). It's a plain text file of `name = value` lines; save it and
    the next command uses it. `fourier config show` checks what it sets."""
    import os
    import shlex
    import subprocess

    from .. import layers
    from ..platforms import open_path
    from ..settings import ConfigError
    explicit = ((ctx.find_root().obj or {}).get("config_args") or (None, ()))[0]
    try:
        path = layers.find_config(explicit)
    except ConfigError as e:
        console.print(f"config: {e}", style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(2) from None
    if path is None:
        console.print("No fourier.toml yet: `fourier setup` writes one "
                      f"({layers.USER_CONFIG}).", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    err = (ctx.find_root().obj or {}).get("config_error")
    if err:
        console.print(f"It doesn't load as it is: {err}", style="yellow", markup=False,
                      highlight=False, soft_wrap=True)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        subprocess.run([*shlex.split(editor), str(path)], check=False)
        console.print(f"{path}: `fourier config show` checks it.", markup=False, highlight=False, soft_wrap=True)
        return
    if open_path(path, text=True) is None:
        console.print(f"No text editor to open it with here: it's {path}.", markup=False,
                      highlight=False, soft_wrap=True)
        raise SystemExit(1)
    console.print(f"Opened {path}. Save it when you're done (Cmd-S on a Mac); the next "
                  "`fourier` command uses it, and `fourier config show` checks it.",
                  markup=False, highlight=False, soft_wrap=True)


def _knob_settings(r) -> list[tuple[str, str, str]]:
    """[(knob, value, file)]: each Tier 1 knob a config file sets (the last file wins, as in
    resolve)."""
    from pathlib import Path

    from .. import layers
    from ..knobs import KNOBS
    got = {}
    for f in r.files:
        p = Path(f)
        try:
            doc = layers._read(p)
        except Exception:
            continue
        if r.config and p == Path(r.config):
            label = f"config:{p.name}"
        elif p.parent == layers.PRESETS_DIR.resolve() or p.parent == layers.PRESETS_DIR:
            label = f"preset:{p.stem}"
        else:
            label = f"file:{p.name}"
        for k in KNOBS:
            if k in doc:
                got[k] = (k, _short(doc[k]), label)
    return [got[k] for k in KNOBS if k in got]


@config_cmd.command("show", short_help="Show the preset, knobs and tunables the config sets.")
@click.option("--all", "show_all", is_flag=True, help="Every tunable, not just the overridden ones")
@click.pass_context
def config_show(ctx, show_all):
    """The preset, the knobs the config files set, and each tunable that differs from the
    code's default with the layer that set it (--all: every tunable)."""
    import os as _os

    from .. import layers
    from ..settings import ENV, decode
    r = (ctx.obj or {}).get("resolved")
    envf = _os.environ.get(ENV)
    if r is None and envf:
        import json as _json
        doc = _json.loads(open(envf).read())
        r = layers.Resolved(values={k: decode(v) for k, v in doc["values"].items()},
                            sources={k: f"${ENV}" for k in doc["values"]}, files=[envf])
    r = r or layers.Resolved()
    say = lambda m: console.print(m, markup=False, highlight=False)   # noqa: E731
    from pathlib import Path as _P

    def _file(f):
        p = _P(f)
        if p.parent in (layers.PRESETS_DIR, layers.PRESETS_DIR.resolve()):
            return f"the {p.stem} style"
        return f
    say(f"config files: {', '.join(_file(f) for f in r.files) or '(none: the code defaults)'}")
    from .setup import preset_line, presets
    if r.preset:
        say(f"preset: {r.preset}" + (f" (the default: {r.config} names none)" if r.preset_default else ""))
    say("presets:")
    names = presets()
    width = max((len(n) for n in names), default=0) + 2
    for n in names:
        say(f"  {n:<{width}}{preset_line(n)}")
    knobs = _knob_settings(r)
    if knobs:
        say("settings (fourier.toml, the style, the overlay):")
        for name, value, where in knobs:
            say(f"  {name} = {value}    ({_source_words(where)})")
    say(_scale_line(r))
    meta = _tunable_meta()
    keys = sorted(meta) if show_all else sorted(r.values)
    if not keys:
        say("tunables: every one at the code's default" + ("" if show_all else " (--all lists them)"))
        return
    say("advanced settings (every one, with its class):" if show_all else
        "advanced settings the style, the knobs or your [advanced] section change:")
    defaults = layers.defaults() if show_all else {}
    for k in keys:
        cls, knob = meta.get(k, ("?", ""))
        v = r.values[k] if k in r.values else defaults.get(k)
        name = k if show_all else k.split(".", 1)[-1]
        src = _source_words(r.sources.get(k, "default"))
        tail = f"; class {cls}" if show_all else ""
        say(f"  {name} = {_short(v, 100)}    ({src}" + (f", the {knob} knob" if knob else "")
            + f"{tail})")


def _source_words(src: str) -> str:
    """config:fourier.toml -> your fourier.toml; preset:balanced -> the balanced style."""
    kind, _, rest = src.partition(":")
    name, _, extra = rest.partition(" ")
    extra = f" {extra}" if extra else ""
    if kind == "config":
        return f"your {name}{extra}"
    if kind == "preset":
        return f"the {name} style{extra}"
    if kind == "file":
        return f"{name}{extra}"
    return src


def _scale_line(r) -> str:
    """What the scale knob resolved to, and what it means for the master's size."""
    from .. import layers
    k = "curate_config.LIBRARY_SCALE"
    on = r.values[k] if k in r.values else layers.defaults()[k]
    per = r.values.get("curate_config.LIBRARY_PER_MASTER", layers.defaults()["curate_config.LIBRARY_PER_MASTER"])
    if on:
        return (f"scale: library (the master follows the library's size and reaches the style's "
                f"budgets at {per} usable samples a file; `fourier doctor` says how big it will be)")
    return "scale: off (the style's budgets whatever the library's size)"


def _explain_knob(name, detail=False) -> None:
    """A Tier 1 knob: what it does, an example line for fourier.toml, and with detail the
    exact rule and the tunables behind it."""
    import inspect

    from .. import layers
    from ..knobs import KNOBS, PLAIN_HELP
    plain, example = PLAIN_HELP.get(name, ("", ""))
    console.print(f"[bold]{name}[/bold]  a setting for fourier.toml (`fourier config edit` opens it)")
    if plain:
        console.print("  " + plain, markup=False, highlight=False)
        console.print(f"  For example:  {example}", markup=False, highlight=False)
    if not detail:
        console.print(f"  `fourier config explain {name} --detail`: the exact rule and the advanced "
                      "settings behind it.", style="dim", markup=False, highlight=False)
        return
    console.print("  " + " ".join(inspect.getdoc(KNOBS[name]).split()), markup=False, highlight=False)  # type: ignore[union-attr]
    known = layers.defaults()
    for k, (cls, knob) in sorted(_tunable_meta().items()):
        if knob == name and k in known:
            console.print(f"  {'sets' if cls == 'knob' else 'uses'} {k} (default {_short(known[k], 80)})",
                          markup=False, highlight=False)


@config_cmd.command("explain", short_help="Explain one setting, or list the category folders.")
@click.argument("key")
@click.option("--detail", is_flag=True, help="For a setting: the exact rule and the advanced settings behind it")
@click.pass_context
def config_explain(ctx, key, detail):
    """One tunable: its class, the knob that sets it, its default and current value. A knob
    (scale, size, tempo, ...): what it does and the tunables behind it. `categories`
    instead: every folder the taxonomy makes, with example files."""
    from .. import layers
    from ..knobs import KNOBS
    from ..settings import ConfigError
    if key == "categories":
        _explain_categories()
        return
    if key in KNOBS:
        _explain_knob(key, detail)
        return
    known = layers.defaults()
    try:
        k = layers.qualify(key, known)
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(2) from None
    cls, knob = _tunable_meta().get(k, ("?", ""))
    r = (ctx.obj or {}).get("resolved") or layers.Resolved()
    console.print(f"[bold]{k}[/bold]  class {cls}" + (f", set by the {knob} knob" if knob else ""))
    console.print(f"  default  {_short(known[k], 200)}")
    if k in r.values:
        console.print(f"  now      {_short(r.values[k], 200)}  (from {r.sources[k]})")
