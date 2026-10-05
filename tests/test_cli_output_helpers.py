"""Words in a fixture's temporary directory are not CLI-message evidence."""

from pathlib import Path

import pytest

from tests.test_first_run import _flat, _without_tmp_path


@pytest.mark.parametrize("root", ["/tmp/render-ready/pytest", r"C:\Temp\render-ready\pytest"])
def test_path_words_are_not_message_words(root):
    output = f"Output: {root}/result.wav"
    assert "render" in output
    assert "render" not in _flat(output, Path(root))
    assert "result.wav" in _flat(output, Path(root))


def test_terminal_wrapping_does_not_restore_false_positive():
    path = Path("/tmp/render-ready/pytest")
    output = "Output: /tmp/ren\nder-ready/py\ntest/file\nActual render completed"
    clean = _without_tmp_path(output, path)
    assert "ren\nder" not in clean
    assert _flat(output, path) == "Output: /file Actual render completed"


def test_real_message_and_unrelated_paths_are_preserved():
    output = "/tmp/current/file render ready /tmp/another/file"
    assert _without_tmp_path(output, Path("/tmp/current")) == (
        "/file render ready /tmp/another/file"
    )
