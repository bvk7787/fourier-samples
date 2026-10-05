"""Network helpers.

Fourier's only calls to a local service (Ollama at localhost, for folder descriptions and
`fourier tools audit`) go through `local_opener()`, which ignores the proxy settings in the
environment (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY and the system's): a request meant for this
machine never leaves it through a proxy, carrying prompts built from the library's file names.
"""
from __future__ import annotations

import urllib.request


def local_opener() -> urllib.request.OpenerDirector:
    """An opener that connects directly, never through a proxy (for localhost services)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))
