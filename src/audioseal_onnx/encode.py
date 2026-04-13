import pathlib
import numpy as np
import onnxruntime as rt
import soxr

SAMPLE_RATE = 16000
MSG_BITS = 16
BATCH_SIZE = 16

_session: rt.InferenceSession | None = None
_session_path: pathlib.Path | None = None


def _get_session(model_path: pathlib.Path) -> rt.InferenceSession:
    global _session, _session_path
    if _session is None or _session_path != model_path:
        _session = rt.InferenceSession(str(model_path))
        _session_path = model_path
    return _session


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    return soxr.resample(samples, orig_sr, target_sr).astype(np.float32)


def add_watermark(samples: np.ndarray, sample_rate: int, model_path: pathlib.Path) -> np.ndarray:
    mono = samples[:, 0] if samples.ndim == 2 else samples
    mono = mono.astype(np.float32)

    audio_16k = _resample(mono, sample_rate, SAMPLE_RATE)
    session = _get_session(model_path)

    chunk = SAMPLE_RATE
    n = len(audio_16k)

    remainder = n % chunk
    if remainder:
        audio_16k = np.concatenate([audio_16k, np.zeros(chunk - remainder, dtype=np.float32)])

    frames = audio_16k.reshape(-1, chunk)
    num_chunks = len(frames)
    msg_batch = np.zeros((BATCH_SIZE, MSG_BITS), dtype=np.float32)

    watermark_chunks = np.zeros_like(frames)
    for i in range(0, num_chunks, BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        actual = len(batch)
        if actual < BATCH_SIZE:
            batch = np.concatenate([batch, np.zeros((BATCH_SIZE - actual, chunk), dtype=np.float32)])
        wm = session.run(None, {
            "audio": batch[:, None, :],
            "message": msg_batch,
        })[0][:, 0, :]
        watermark_chunks[i:i + actual] = wm[:actual]

    watermark_16k = watermark_chunks.reshape(-1)[:n]
    watermark = _resample(watermark_16k, SAMPLE_RATE, sample_rate)[:len(mono)]

    result = mono + watermark
    if samples.ndim == 2:
        out = samples.copy().astype(np.float32)
        out[:, 0] = result
        return out
    return result
