# audioseal-onnx

Lightweight audio watermarking powered by [AudioSeal](https://github.com/facebookresearch/audioseal)

- No PyTorch — runs on `onnxruntime` only (~34 MB INT8 models)
- Any sample rate and any length
- 10 min detected in ~5s on CPU
- Survives MP3 (320k–64k), AAC, resampling, volume changes, and trimming

## Install

```console
uv pip install git+https://github.com/thewh1teagle/audioseal-onnx
wget https://github.com/thewh1teagle/audioseal-onnx/releases/download/v1.0.0/models.tar.gz
tar xf models.tar.gz
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

## Model Export

The ONNX models are pre-exported and included. To re-export from AudioSeal:

```console
uv run scripts/export_onnx.py
```

## Benchmark

```console
uv run scripts/benchmark.py
```
