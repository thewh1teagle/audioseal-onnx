# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "audioseal-onnx @ file:///${PROJECT_ROOT}",
#   "soundfile==0.13.1",
# ]
# ///
"""
Benchmark watermark detection after various ffmpeg compressions.

Usage:
    uv run scripts/benchmark.py
"""

import subprocess
import tempfile
import pathlib
import soundfile as sf
from audioseal_onnx import add_watermark, detect_watermark

ROOT = pathlib.Path(__file__).parent.parent
INPUT = ROOT / "input.wav"
ENCODER_MODEL = ROOT / "models" / "encoder.onnx"
DECODER_MODEL = ROOT / "models" / "decoder.onnx"

SCENARIOS = [
    ("WAV (original)",        ["ffmpeg", "-i", "{src}", "-c:a", "pcm_s16le", "{dst}"]),
    ("MP3 320k",              ["ffmpeg", "-i", "{src}", "-c:a", "libmp3lame", "-b:a", "320k", "{dst}"]),
    ("MP3 128k",              ["ffmpeg", "-i", "{src}", "-c:a", "libmp3lame", "-b:a", "128k", "{dst}"]),
    ("MP3 64k",               ["ffmpeg", "-i", "{src}", "-c:a", "libmp3lame", "-b:a", "64k", "{dst}"]),
    ("AAC 128k",              ["ffmpeg", "-i", "{src}", "-c:a", "aac", "-b:a", "128k", "-f", "mp4", "{dst}"]),
    ("OGG Vorbis q5",         ["ffmpeg", "-i", "{src}", "-c:a", "libvorbis", "-q:a", "5", "{dst}"]),
    ("Resample 8kHz",         ["ffmpeg", "-i", "{src}", "-ar", "8000", "{dst}"]),
    ("Resample 22kHz",        ["ffmpeg", "-i", "{src}", "-ar", "22050", "{dst}"]),
    ("Volume +6dB",           ["ffmpeg", "-i", "{src}", "-af", "volume=6dB", "{dst}"]),
    ("Volume -6dB",           ["ffmpeg", "-i", "{src}", "-af", "volume=-6dB", "{dst}"]),
    ("Trim first 50%",        ["ffmpeg", "-i", "{src}", "-ss", "0", "-t", "{half}", "{dst}"]),
]

def run_ffmpeg(cmd: list[str], src: pathlib.Path, dst: pathlib.Path, half: str = "0"):
    cmd = [c.replace("{src}", str(src)).replace("{dst}", str(dst)).replace("{half}", half) for c in cmd]
    subprocess.run(cmd, capture_output=True, check=True)

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        # Read and watermark
        samples, sr = sf.read(INPUT)
        duration = len(samples) / sr
        half = str(round(duration / 2, 2))

        print("Watermarking input.wav...")
        watermarked, sr = add_watermark(samples, sr, encoder_model=ENCODER_MODEL)
        watermarked_path = tmp / "watermarked.wav"
        sf.write(watermarked_path, watermarked, sr)

        print(f"\n{'Scenario':<25} {'Detected':<10}")
        print("-" * 35)

        for name, cmd in SCENARIOS:
            suffix = pathlib.Path(cmd[-1].replace("{dst}", "x")).suffix or ".wav"
            # Determine output suffix from ffmpeg command
            if "mp3" in " ".join(cmd) or "libmp3lame" in " ".join(cmd):
                suffix = ".mp3"
            elif "aac" in " ".join(cmd):
                suffix = ".m4a"
            elif "vorbis" in " ".join(cmd):
                suffix = ".ogg"
            else:
                suffix = ".wav"

            out = tmp / f"out{suffix}"
            try:
                run_ffmpeg(cmd, watermarked_path, out, half=half)
                # Decode back to WAV if soundfile can't read the format directly
                if out.suffix != ".wav":
                    decoded = tmp / "decoded.wav"
                    subprocess.run(["ffmpeg", "-y", "-i", str(out), str(decoded)], capture_output=True, check=True)
                    read_path = decoded
                else:
                    read_path = out
                result_samples, result_sr = sf.read(read_path)
                detected = detect_watermark(result_samples, result_sr, decoder_model=DECODER_MODEL)
                status = "✓" if detected else "✗"
            except subprocess.CalledProcessError:
                status = "SKIP (codec unavailable)"
            except Exception as e:
                status = f"ERROR: {e}"

            print(f"{name:<25} {status}")

if __name__ == "__main__":
    main()
