# Baselines

| Directory | Baseline | Code |
| --- | --- | --- |
| [`ar/`](ar/) | HF autoregressive decoding (AR) | `decoding.py` |
| [`sd/`](sd/) | Standard speculative decoding (SD) | `decoding.py` |
| [`fsd/`](fsd/) | Fuzzy speculative decoding (FSD) | `decoding.py`, `generate_fsd.py` |
| [`pipespec/`](pipespec/) | PipeSpec | `decoding.py`, edge/cloud RTT entrypoints and transport |
| [`specedge/`](specedge/) | SpecEdge | `src/`, with DeepSeek/Gemma variant patches |

AR, SD and FSD retain the recorded `ARM` settings and `run_one` interface.
Call `configure(target, draft)` before SD/FSD. Their shared generation core is
in `fsd/generate_fsd.py`; the original environment used PyTorch 2.11.0 and
Transformers 4.57.1. Import these modules from the repository root.

Third-party licenses are retained alongside the code.
Models and input data are supplied separately.

For SpecEdge, apply either `deepseek.patch` or `gemma.patch` to a separate copy
of `specedge/`.
