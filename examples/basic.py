"""
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O input.wav
wget https://github.com/thewh1teagle/audioseal-onnx/releases/download/v1.0.0/models.tar.gz
tar xf models.tar.gz
uv run examples/basic.py
"""

input_path = "input.wav"
output_path = "marked.wav"

from audioseal_onnx import add_watermark, detect_watermark
import soundfile as sf

# Read the input audio file
samples, sr = sf.read(input_path)
# Watermark the file
samples, sr = add_watermark(samples, sr)

# Write the output audio file
sf.write(output_path, samples, sr)
print(f"Watermarked audio saved to: {output_path}")

# Detect the watermark

print(f"Detecting watermark in: {output_path}")
samples, sr = sf.read(output_path)
is_watermarked = detect_watermark(samples, sr)
print(f"Watermark detected: {is_watermarked}")
