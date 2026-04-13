# audioseal-onnx

Lightweight Python library for audio watermarking powered by [AudioSeal](https://github.com/facebookresearch/audioseal) (Meta), exported to ONNX for inference without PyTorch.

## Install

```bash
uv pip install git+https://github.com/thewh1teagle/audioseal-onnx
```

## Usage

See [`examples/basic.py`](examples/basic.py) for a full example.

```python
import soundfile as sf
from audioseal_onnx import add_watermark, detect_watermark

samples, sr = sf.read("audio.wav")

# Add watermark
watermarked, sr = add_watermark(samples, sr)
sf.write("watermarked.wav", watermarked, sr)

# Detect watermark
is_watermarked = detect_watermark(watermarked, sr)
print(f"Watermark detected: {is_watermarked}")
```

## Features

- **Lightweight** — no PyTorch at runtime, just `onnxruntime` + `numpy` + `soxr`
- **Small models** — INT8 quantized: encoder 21 MB, decoder 12 MB (total ~34 MB)
- **Any sample rate** — audio is resampled to 16 kHz internally and back, transparent to the user
- **Any length** — works on short clips or multi-hour recordings
- **Fast detection** — 10 minutes of audio detected in ~5s on CPU
- **Robust** — survives MP3 (320k–64k), AAC, resampling (8–44kHz), volume changes, and trimming

## Model Export

The ONNX models are pre-exported and included. To re-export from AudioSeal:

```bash
uv run scripts/export_onnx.py
```
