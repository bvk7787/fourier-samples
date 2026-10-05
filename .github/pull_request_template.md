## What this changes

<!-- What and why. Link the issue if there is one. -->

## Checks

- [ ] Tests run (`pytest -q -n auto tests`), and a behavior change comes with a test
- [ ] For a curation change: the synthetic golden (`python tests/golden/synthetic_build.py /tmp/syn --check`); if the output is meant to change, regenerated with `--update` and explained above
- [ ] For a command, option or help-text change: `python tests/test_cli_tree.py --write`
- [ ] `ruff check src tests`
- [ ] `mypy`
- [ ] Docs updated (README, CLAUDE.md, docs/) where the behavior they describe changed
- [ ] No personal data and no real samples: fixtures use generated audio and invented names (Acme, Northwind, ...), and no real vendor, pack or person is named
- [ ] For a device profile: every value cites a manual page or is marked a convention, and what I checked on the hardware is described above
