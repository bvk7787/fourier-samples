"""
fourier CLI

All commands print structured, human-readable output. `fourier --help` lists them in sections
(cli/_app.py, SECTIONS):

  Get started        fourier setup | demo | doctor
  Build and load     fourier build [--all] | render <device> | sync <device> <volume>
  Releases           fourier publish | releases
  Look inside        fourier open | why | verify | diff | search
  Settings           fourier config edit|show|explain | devices list|show|new
  Listen and rate    fourier review rate|import|queue|score|misfiles|ratings
  Advanced           fourier tools scan|analyze|audit|dedup|import-folder|resolve|db-stats
"""

from __future__ import annotations

# the command modules register their commands on main (and on tools) when imported
from . import build, config, db, devices, enrich, ingest, opener, releases, review, search, setup  # noqa: F401
from ._app import _table_cols, console, main  # noqa: F401
from .build import _run_verify, _swap_in  # noqa: F401
from .releases import _release_problems, _release_staging  # noqa: F401

__all__ = ["main"]
