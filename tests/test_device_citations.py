"""Device profiles cite the manual: every quote must be on its cited page of the current manual.

This guards against device facts written from memory and flags citations that a new manual
version moved.
Every value in a profile says where it comes from (a citation, or convention/unverified);
the loader refuses a value that doesn't.

The page check reads the manual PDFs from $FOURIER_MANUALS_DIR (any file names: each profile's
manual is found by its sha256) and skips when the variable is unset or no file there matches.
"""
import hashlib
import os
import re
import unicodedata
from pathlib import Path

import pytest
import yaml

from fourier.devices.loader import DEFAULT_DEVICES_DIR, DeviceLoader, DeviceProfile, DeviceProfileError

PROFILES = sorted(p for p in DEFAULT_DEVICES_DIR.glob("*.yaml") if not p.name.startswith("."))
VERIFIED = [p for p in PROFILES if yaml.safe_load(p.read_text()).get("status") != "unverified"]

_norm_re = re.compile(r"[^0-9a-z]")


def norm(text: str) -> str:
    """Lowercase alphanumerics only, after NFKC (so ligatures like U+FB01 compare as 'fi')."""
    return _norm_re.sub("", unicodedata.normalize("NFKC", text).lower())


def _raw(path):
    return yaml.safe_load(path.read_text())


def _manual_pdf(sha256: str) -> Path | None:
    """The PDF in $FOURIER_MANUALS_DIR whose sha256 is the profile's, whatever its name."""
    folder = os.environ.get("FOURIER_MANUALS_DIR")
    if not folder:
        return None
    for f in sorted(Path(folder).expanduser().rglob("*")):
        if f.is_file() and f.suffix.lower() == ".pdf":
            if hashlib.sha256(f.read_bytes()).hexdigest() == sha256:
                return f
    return None


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_profile_loads_and_every_value_has_a_source(path):
    p = DeviceProfile.from_yaml(path)
    raw = _raw(path)
    given = [f"{sec}.{k}" for sec in ("paths", "audio") for k in (raw.get(sec) or {})]
    given += ["storage_mb"] if "storage_mb" in raw else []
    assert sorted(p.sources) == sorted(given)
    for name in given:
        assert p.source(name) != "default"


@pytest.mark.parametrize("path", VERIFIED, ids=lambda p: p.stem)
def test_profile_names_its_manual_and_cites_it(path):
    raw = _raw(path)
    manual = raw.get("manual")
    assert isinstance(manual, dict), f"{path.name}: manual must be {{title, version, url, sha256}}"
    assert manual.get("title") and manual.get("version"), path.name
    assert str(manual.get("url", "")).startswith("https://"), path.name
    assert re.fullmatch(r"[0-9a-f]{64}", str(manual.get("sha256", ""))), path.name
    cites = raw.get("citations") or {}
    assert len(cites) >= 5
    for c in cites.values():
        assert c.get("claim") and isinstance(c.get("page"), int) and len(norm(c.get("quote", ""))) >= 12


@pytest.mark.parametrize("path", VERIFIED, ids=lambda p: p.stem)
def test_citation_quotes_are_on_their_pages(path):
    raw = _raw(path)
    if not os.environ.get("FOURIER_MANUALS_DIR"):
        pytest.skip("FOURIER_MANUALS_DIR not set (the folder holding the manual PDFs)")
    pdf_path = _manual_pdf(raw["manual"]["sha256"])
    if pdf_path is None:
        pytest.skip(f"{path.name}: no PDF in $FOURIER_MANUALS_DIR matches the manual's sha256")
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        wrong = []
        for c in raw["citations"].values():
            # page: is the PDF page (1-based)
            page = norm(pdf[c["page"] - 1].get_textpage().get_text_range())
            if norm(c["quote"]) not in page:
                wrong.append((c["page"], c["quote"][:60]))
    finally:
        pdf.close()
    assert not wrong, f"{path.name}: quotes not found on cited pages: {wrong}"


def test_generic_profiles_claim_nothing():
    """The fallbacks for devices without a profile cite no manual and set no limits."""
    for did in ("generic_44k", "generic_48k"):
        p = DeviceLoader().load(did)
        assert not p.verified and p.manual is None and not p.citations
        assert p.max_path_length is None and p.storage_mb is None and p.card_dir == ""
        assert {p.source(n) for n in p.sources} == {"unverified"}


def test_a_value_without_a_source_is_refused():
    raw = {"id": "x", "audio": {"sample_rate": {"value": 48000}}}
    with pytest.raises(DeviceProfileError, match="cite"):
        DeviceProfile.from_dict(raw)
    raw["audio"]["sample_rate"]["cite"] = "nowhere"
    with pytest.raises(DeviceProfileError, match="no citation"):
        DeviceProfile.from_dict(raw)
    with pytest.raises(DeviceProfileError, match="unknown keys"):
        DeviceProfile.from_dict({"id": "x", "constraints": {}})


def test_the_manual_is_a_title_version_url_and_sha256():
    good = {"title": "My Sampler Manual", "version": "1.0", "url": "https://example.com/m.pdf",
            "sha256": "0" * 64}
    cites = {"rate": {"claim": "48 kHz", "page": 3, "quote": "plays 48 kHz audio files"}}
    raw = {"id": "x", "manual": good, "citations": cites,
           "audio": {"sample_rate": {"value": 48000, "cite": "rate"}}}
    assert DeviceProfile.from_dict(raw).manual == good
    assert "My Sampler Manual 1.0" in DeviceProfile.from_dict(raw).summary()
    with pytest.raises(DeviceProfileError, match="manual"):
        DeviceProfile.from_dict({**raw, "manual": "my_sampler_manual"})
    with pytest.raises(DeviceProfileError, match="manual"):
        DeviceProfile.from_dict({**raw, "manual": {k: v for k, v in good.items() if k != "url"}})
    with pytest.raises(DeviceProfileError, match="sha256"):
        DeviceProfile.from_dict({**raw, "manual": {**good, "sha256": "abc"}})
    with pytest.raises(DeviceProfileError, match="citations need"):
        DeviceProfile.from_dict({k: v for k, v in raw.items() if k != "manual"})


def test_the_manual_is_found_by_its_sha256(tmp_path, monkeypatch):
    pdf = tmp_path / "any name.pdf"
    pdf.write_bytes(b"%PDF-1.4 stand-in")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.delenv("FOURIER_MANUALS_DIR", raising=False)
    assert _manual_pdf(sha) is None
    monkeypatch.setenv("FOURIER_MANUALS_DIR", str(tmp_path))
    assert _manual_pdf(sha) == pdf
    assert _manual_pdf("0" * 64) is None


def test_known_bad_claims_stay_out():
    """Claims the manuals contradict must not creep back into the profiles."""
    m8 = (DEFAULT_DEVICES_DIR / "m8_tracker.yaml").read_text()
    assert DeviceLoader().load("m8_tracker").formats == ["wav"]
    assert "WAV/AIFF" not in m8
