# mflux-streamlit

[![PyPI - Version](https://img.shields.io/pypi/v/mflux-streamlit.svg)](https://pypi.org/project/mflux-streamlit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mflux-streamlit.svg)](https://pypi.org/project/mflux-streamlit)

A web app gui for [mflux](https://pypi.org/project/mflux/) python library implemented with [Streamlit](https://docs.streamlit.io/)

-----

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [License](#license)

## Features

- Image generation usig mflux along with ControlNet Support
- Multiple LoRAs and scales for each LoRA adapter
- ControlNet integration for enhanced control

## Previews
| ![Left: FLUX.1](images/flux.png) | ![Right: FLUX.1 with LoRA](images/flux-lora.png) |
| --- | --- |
| FLUX.1 Schnell | FLUX.1 Schnell with LoRA |

| ![Left: FLUX.1](images/pose.png)| ![Right: FLUX.1 with LoRA](images/flux-controlnet.png) |
| --- | --- |
| ControlNet Image<br/>[Source](https://atlegras.medium.com/pose-like-a-pro-ais-recommendations-for-woman-standing-portraits-1c4194ae63c6) | FLUX.1 Schnell with Controlnet |



### Todo List

- Beautify the web UI and improve UX
- support `mflux`'s existing Image-to-Image modes

## Installation

### Normal User Install

```sh
brew install uv
uv tool install mflux-streamlit
cd /your/directory

mflux-streamlit  
```

### Developer Install

```sh
# dev workaround: run the main.py
uv venv && source .venv/bin/activate
uv pip install -e .
streamlit src/app.py
```

## License

`mflux-streamlit` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.