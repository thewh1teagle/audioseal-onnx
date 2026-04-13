# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "audioseal==0.2.0",
#   "torch==2.11.0",
#   "onnx==1.21.0",
#   "onnxruntime==1.24.4",
# ]
# ///
"""
Export AudioSeal generator and detector to ONNX format (INT8 quantized).

Run once:
    uv run scripts/export_onnx.py

Output:
    models/encoder.onnx
    models/decoder.onnx
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import pathlib
import torch
from audioseal import AudioSeal
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

GENERATOR_PATH = MODELS_DIR / "encoder.onnx"
DETECTOR_PATH = MODELS_DIR / "decoder.onnx"
MSG_BITS = 16

torch._dynamo.config.disable = True

print("Loading generator...")
generator = AudioSeal.load_generator("audioseal_wm_16bits")
generator.eval()
generator.normalizer = None
generator = generator.__prepare_scriptable__()

print("Loading detector...")
detector = AudioSeal.load_detector("audioseal_detector_16bits")
detector.eval()
detector = detector.__prepare_scriptable__()

dummy_audio = torch.randn(1, 1, 16000)
dummy_msg = torch.randint(0, 2, (1, MSG_BITS), dtype=torch.float32)


class GeneratorWrapper(torch.nn.Module):
    """Wrap generator to fix sample_rate tracing issue — always assumes 16kHz input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return self.model.get_watermark(audio, sample_rate=None, message=message)


print(f"Exporting generator to {GENERATOR_PATH}...")
torch.onnx.export(
    GeneratorWrapper(generator),
    (dummy_audio, dummy_msg),
    str(GENERATOR_PATH),
    input_names=["audio", "message"],
    output_names=["watermark"],
    dynamic_axes={"audio": {0: "batch", 2: "frames"}, "message": {0: "batch"}, "watermark": {0: "batch", 2: "frames"}},
    opset_version=17,
    dynamo=False,
)

print(f"Exporting detector to {DETECTOR_PATH}...")
torch.onnx.export(
    detector,
    (dummy_audio,),
    str(DETECTOR_PATH),
    input_names=["audio"],
    output_names=["result"],
    dynamic_axes={"audio": {0: "batch", 2: "frames"}, "result": {0: "batch", 2: "frames"}},
    opset_version=17,
    dynamo=False,
)

print("Quantizing to INT8...")
fp32_encoder = MODELS_DIR / "encoder_fp32.onnx"
fp32_decoder = MODELS_DIR / "decoder_fp32.onnx"
GENERATOR_PATH.rename(fp32_encoder)
DETECTOR_PATH.rename(fp32_decoder)
quantize_dynamic(fp32_encoder, GENERATOR_PATH, weight_type=QuantType.QInt8)
quantize_dynamic(fp32_decoder, DETECTOR_PATH, weight_type=QuantType.QInt8)
fp32_encoder.unlink()
fp32_decoder.unlink()

encoder_mb = GENERATOR_PATH.stat().st_size / 1024 / 1024
decoder_mb = DETECTOR_PATH.stat().st_size / 1024 / 1024
print(f"\nDone.")
print(f"  encoder.onnx: {encoder_mb:.1f} MB")
print(f"  decoder.onnx: {decoder_mb:.1f} MB")
