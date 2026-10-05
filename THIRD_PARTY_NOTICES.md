# Fourier Samples: third-party notices

Fourier Samples is Apache-2.0. It uses, but does not include, the following. Each is installed
or downloaded separately and keeps its own license.

## CLAP model (the `[clap]` extra)

Fourier Samples' CLAP embeddings come from `laion/clap-htsat-unfused` on Hugging Face, pinned to
revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` (`src/fourier/analysis/clap_features.py`)
and downloaded on first use. The model card lists the weights under the Apache-2.0 license
(https://huggingface.co/laion/clap-htsat-unfused).

The model was trained on LAION-Audio-630K, about 633,000 audio-text pairs gathered from
several sources. The dataset's README says the audio may be used for research purposes only
unless the owners of a source give permission for other uses
(https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md). The
dataset's terms are its own; Fourier Samples does not download or ship the dataset, its audio
or the model's weights.

Fourier Samples ships no sound-model weights. Its sound model (`src/fourier/metadata/sound.py`)
is trained on each user's own machine, from their own library (`fourier tools train`, or a
build without Sononym), and its weights stay in their Fourier home: they are learned from that
library's samples, whose licenses are the user's.

## Rubber Band (the `[stretch]` extra)

Time-stretching can use `pyrubberband` (ISC license), which runs the `rubberband`
command-line program as a separate process (https://github.com/bmcfee/pyrubberband). The
Rubber Band Library is GPL-2.0-or-later, with a commercial license available from Breakfast
Quay (https://breakfastquay.com/rubberband/license.html). Fourier Samples doesn't link it and
falls back to librosa when it isn't installed.

## Device manuals

Device profiles (`config/devices/*.yaml`) name the manual they cite (title, version, its
maker's URL and the PDF's sha256) and quote short passages from it, each with its page, as
citations for the values they set. The manuals are their makers' copyright; Fourier Samples
neither downloads nor includes them.

## Folder descriptions (optional)

With folder descriptions on (`fourier setup --llm`, which sets `DESCRIBE`), Fourier Samples asks
a local model through Ollama's HTTP API (default `qwen3.5:9b`, Apache-2.0). You install Ollama
and pull the model yourself.

## Libraries

Fourier Samples' Python dependencies (`pyproject.toml`, `uv.lock`) are installed from PyPI under
their own licenses, all permissive or weak copyleft, including: LGPL `soxr` and the libsndfile,
mpg123 and LAME libraries bundled in `soundfile`'s wheels; MPL-2.0 `certifi` and `tqdm`; and,
with the `[dev]` extra (the citation test), PDFium and its dependencies, bundled in `pypdfium2`
(itself Apache-2.0 / BSD-3-Clause) under BSD-style and other permissive terms. Anyone bundling
Fourier Samples with its dependencies (an app bundle, a container image) takes on those terms, and
on Linux, NVIDIA's for the CUDA libraries PyTorch pulls in.

## Trademarks

Elektron, Digitakt, Dirtywave, M8, Ableton, Live, Sononym and the other product and company
names in this repository, including instrument names used to label folders, are trademarks of
their owners. Fourier Samples is an independent project, not affiliated with or endorsed by any
of them.
