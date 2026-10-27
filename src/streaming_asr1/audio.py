import io
import wave

import numpy as np


def rms_dbfs(samples: np.ndarray) -> float:
    """float32([-1, 1]) 샘플의 RMS 레벨(dBFS). 무음이면 -100."""
    if samples.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return 20.0 * np.log10(rms) if rms > 1e-5 else -100.0


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """PCM WAV 를 mono float32 와 샘플레이트로 디코딩한다."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"PCM WAV 파일만 지원합니다: {e}") from e

    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8
        x = i32.astype(np.float32) / 8388608.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 샘플 폭: {width}")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """선형 보간 리샘플링 (브라우저 클라이언트의 워클릿과 같은 방식)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    n = int(round(samples.size * dst_rate / src_rate))
    return np.interp(np.arange(n) * (src_rate / dst_rate), np.arange(samples.size), samples).astype(np.float32)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
