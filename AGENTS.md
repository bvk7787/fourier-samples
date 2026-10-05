# Instructions for coding agents

Helping someone *use* Fourier Samples (their library, a build, a device)? Read
[skills/fourier/SKILL.md](skills/fourier/SKILL.md). Changing its code? Read on.

- Read [CLAUDE.md](CLAUDE.md) first: the ground rules (the library's files are never changed,
  device facts come from the manuals, no one's library in the code), the architecture, and the
  checks to run before a change is done.
- Adding or changing a device profile: [docs/device-profiles.md](docs/device-profiles.md).
- Contributing (tests, the CLI snapshot, what a pull request needs):
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Ratings don't need Ableton Live: `fourier review rate` and `fourier review import` write the
  same store Live's tag harvest writes (packs/ratings.py).
