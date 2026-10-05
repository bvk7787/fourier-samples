"""Fourier Samples: curate a sample library into device-ready folders for hardware samplers."""
from __future__ import annotations

import os as _os
import warnings as _warnings

# librosa's and numba's warnings about a short or quiet input ("n_fft=1024 is too large for
# input signal of length=...", "Trying to estimate tuning from empty frequency set") are
# expected on one-shots and say nothing a user can act on, so they stay out of a build's
# and an analysis' output, in this process and in every worker (each imports this package).
# Errors are exceptions, never warnings, and stay visible; `fourier -v` (FOURIER_VERBOSE=1)
# shows the warnings too.
VERBOSE_ENV = "FOURIER_VERBOSE"
# what to do about a missing core dependency (duckdb, librosa, soundfile: every install has
# them, so one is missing only from a broken install)
REINSTALL = ("it comes with Fourier Samples, so the install is incomplete: reinstall it (the "
             "README's Updating section), then run `fourier doctor`")
_QUIET = ((UserWarning, r"librosa(\.|$)"), (Warning, r"numba(\.|$)"))


def quiet_library_warnings(on: bool = True) -> None:
    """Hide (on) or show again (off) librosa's and numba's warnings in this process."""
    for category, module in _QUIET:
        _warnings.filterwarnings("ignore" if on else "default", category=category, module=module)


if not _os.environ.get(VERBOSE_ENV):
    quiet_library_warnings(True)
