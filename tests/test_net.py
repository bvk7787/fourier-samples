"""Calls to a local service never go through a proxy (fourier/net.py)."""
import urllib.request

from fourier.net import local_opener


def test_the_local_opener_ignores_proxy_settings(monkeypatch):
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(k, "http://proxy.invalid:3128")
    proxied = [h for h in urllib.request.build_opener().handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert proxied and proxied[0].proxies.get("http")          # the default opener would use it
    assert not [h for h in local_opener().handlers if isinstance(h, urllib.request.ProxyHandler)]
