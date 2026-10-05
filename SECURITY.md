# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

Fixes go into the latest 0.1.x release.

## What Fourier Samples does

Fourier Samples is a command-line tool that runs on your own machine. It runs no server, opens
no listening port and sends no telemetry. It reads your sample library and writes only:

- its own folders: its home (`~/.fourier` by default, with its logs and report pages), its
  config (`~/.config/fourier`), and the master, renders and releases;
- the CLAP model in the Hugging Face cache (`~/.cache/huggingface`);
- the device card you name to `fourier sync`;
- the copies `fourier tools import-folder` adds to a library folder.

It uses the network only for:

- `fourier setup` installing PyTorch and Transformers when you say yes (from PyPI, or PyTorch's
  own index for a CPU-only build), and offering `uv self update` when uv is too old for that;
- downloading the CLAP model from Hugging Face once, pinned to one revision;
- a local Ollama server on `localhost`, when you turn folder descriptions on or run `fourier
  tools audit`.

The report page (`fourier open report`) is a local file that plays the master's own files. It
loads nothing from the network.

Issues worth reporting privately include:

- a way for a crafted audio file, config file, device profile or plug-in setting to run code,
  read or write outside the folders above, or delete or change files in a library;
- a path a build, render, publish or sync would write that escapes its target folder.

## Reporting a vulnerability

Please report it privately through GitHub: on the repository's **Security** tab, choose
**Report a vulnerability** (a private security advisory). Don't open a public issue for it.
Include the version (`fourier --version`), your OS and Python version, and the steps or files
that show the problem.

You can expect an acknowledgment within a week. We'll work on a fix in the private advisory,
credit you in the release notes unless you'd rather not be named, and publish the advisory when
a fixed release is out.
