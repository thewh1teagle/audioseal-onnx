import pathlib
from .encode import add_watermark as _add_watermark
from .decode import detect_watermark as _detect_watermark

DEFAULT_THRESHOLD = 0.5
DEFAULT_ENCODER_MODEL = pathlib.Path("models/encoder.onnx")
DEFAULT_DECODER_MODEL = pathlib.Path("models/decoder.onnx")


def add_watermark(
    samples,
    sample_rate: int,
    encoder_model: pathlib.Path = DEFAULT_ENCODER_MODEL,
) -> tuple:
    """
    Add a watermark to the audio samples.

    Args:
        samples: The audio samples to be watermarked.
        sample_rate (int): The sample rate of the audio.
        encoder_model (Path): Path to the encoder ONNX model.

    Returns:
        tuple: A tuple containing the watermarked audio samples and the sample rate.
    """
    return _add_watermark(samples, sample_rate, encoder_model), sample_rate


def detect_watermark(
    samples,
    sample_rate: int,
    threshold: float = DEFAULT_THRESHOLD,
    decoder_model: pathlib.Path = DEFAULT_DECODER_MODEL,
) -> bool:
    """
    Detect whether a watermark is present in the audio samples.

    Args:
        samples: The audio samples to check.
        sample_rate (int): The sample rate of the audio.
        threshold (float): Detection threshold (0.0–1.0).
        decoder_model (Path): Path to the decoder ONNX model.

    Returns:
        bool: True if the watermark is detected, False otherwise.
    """
    return _detect_watermark(samples, sample_rate, threshold, decoder_model)
