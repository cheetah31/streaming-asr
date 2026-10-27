"""NeMo-Speech.cpp 의 안정 C ABI(include/nemo_speech/asr.h)를 ctypes 로 감싼 래퍼.

FastAPI 프로세스 안에서 GGUF 모델을 직접 로드해 오프라인/스트리밍 인식을 수행한다.
구조체 정의는 NeMo-Speech.cpp v0.2.0 의 asr.h 와 1:1 로 대응한다.
"""

from __future__ import annotations

import ctypes as C
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- C 구조체 (asr.h) ----


class _BackendConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("gpu", C.c_int32)]


class _ModelConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("path", C.c_char_p), ("name", C.c_char_p)]


class _StreamingConfig(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("chunk_size", C.c_float),
        ("ctc_left_padding", C.c_float),
        ("ctc_right_padding", C.c_float),
        ("rnnt_right_context", C.c_int32),
    ]


class _RecognizerConfig(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("backend", C.POINTER(_BackendConfig)),
        ("model", C.POINTER(_ModelConfig)),
        ("streaming", C.POINTER(_StreamingConfig)),
        ("decoder", C.c_void_p),
        ("vad", C.c_void_p),
        ("endpointing", C.c_void_p),
        ("postproc", C.c_void_p),
        ("diar", C.c_void_p),
        ("batching", C.c_void_p),
    ]


class _SpeechContext(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("phrases", C.POINTER(C.c_char_p)),
        ("phrase_count", C.c_size_t),
        ("boost", C.c_float),
    ]


class _RecognitionOptions(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("request_id", C.c_char_p),
        ("language_code", C.c_char_p),
        ("interim_results", C.c_bool),
        ("enable_word_time_offsets", C.c_bool),
        ("enable_automatic_punctuation", C.c_bool),
        ("verbatim_transcripts", C.c_bool),
        ("profanity_filter", C.c_bool),
        ("stop_history_eou_ms", C.c_int32),
        ("speech_contexts", C.POINTER(_SpeechContext)),
        ("speech_context_count", C.c_size_t),
        ("max_alternatives", C.c_int32),
        ("enable_speaker_diarization", C.c_bool),
        ("max_speaker_count", C.c_int32),
    ]


_OK = 0
_LANG_TAG = re.compile(r"\s*<([a-z]{2}-[A-Z]{2})>\s*")
_f32p = C.POINTER(C.c_float)
_vp = C.c_void_p


class NemoASRError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechContext:
    """인식 시 우선시할 단어/구문 묶음 (word boosting). RNNT 모델은 boost 2~3 이 일반적이며 최대 5.0 이다."""

    phrases: tuple[str, ...]
    boost: float


@dataclass
class ASRResult:
    text: str
    is_final: bool
    audio_processed: float  # 스트림 시작부터 처리된 오디오 길이(초)
    language: str | None


def _load_library(lib_dir: str | os.PathLike) -> C.CDLL:
    lib_dir = Path(lib_dir).resolve()
    if sys.platform == "win32":
        # 의존 DLL(ggml-*, cublas, msvcp ...)을 같은 bin 폴더에서 찾도록 한다.
        os.add_dll_directory(str(lib_dir))
        os.environ["PATH"] = f"{lib_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        name = "nemo_speech_asr_c.dll"
    elif sys.platform == "darwin":
        name = "libnemo_speech_asr_c.dylib"
    else:
        name = "libnemo_speech_asr_c.so"
    path = lib_dir / name
    if not path.exists():
        raise NemoASRError(f"NeMo-Speech 라이브러리를 찾을 수 없습니다: {path} (uv run streaming-asr1-setup 으로 설치)")
    lib = C.CDLL(str(path))

    def sig(fn: str, restype, *argtypes):
        f = getattr(lib, fn)
        f.restype = restype
        f.argtypes = list(argtypes)

    sig("nemo_speech_asr_recognition_options_default", _RecognitionOptions)
    sig("nemo_speech_asr_create", C.c_int, C.POINTER(_RecognizerConfig), C.POINTER(_vp))
    sig("nemo_speech_asr_destroy", None, _vp)
    sig("nemo_speech_asr_recognize_f32", C.c_int, _vp, C.POINTER(_RecognitionOptions), _f32p, C.c_size_t, C.c_int32, C.POINTER(_vp))
    sig("nemo_speech_asr_streaming_recognize", C.c_int, _vp, C.POINTER(_RecognitionOptions), C.POINTER(_vp))
    sig("nemo_speech_asr_stream_push_f32", C.c_int, _vp, _f32p, C.c_size_t, C.c_int32)
    sig("nemo_speech_asr_stream_force_endpoint", C.c_int, _vp)
    sig("nemo_speech_asr_stream_finish", C.c_int, _vp)
    sig("nemo_speech_asr_stream_next", C.c_int, _vp, C.POINTER(_vp))
    sig("nemo_speech_asr_stream_close", None, _vp)
    sig("nemo_speech_asr_result_is_final", C.c_bool, _vp)
    sig("nemo_speech_asr_result_audio_processed", C.c_float, _vp)
    sig("nemo_speech_asr_result_alternative_count", C.c_size_t, _vp)
    sig("nemo_speech_asr_result_transcript", C.c_char_p, _vp, C.c_size_t)
    sig("nemo_speech_asr_result_language_count", C.c_size_t, _vp, C.c_size_t)
    sig("nemo_speech_asr_result_language_code", C.c_char_p, _vp, C.c_size_t, C.c_size_t)
    sig("nemo_speech_asr_result_destroy", None, _vp)
    sig("nemo_speech_asr_last_error", C.c_char_p)
    sig("nemo_speech_asr_version", C.c_char_p)
    return lib


class NemoRecognizer:
    """GGUF ASR 모델 하나를 로드한다. 여러 스레드에서 동시에 호출해도 된다."""

    def __init__(self, lib_dir: str | os.PathLike, model_path: str | os.PathLike, gpu: int = 0, right_context: int = 1):
        if not Path(model_path).is_file():
            raise NemoASRError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        self._lib = _load_library(lib_dir)
        self.model_path = str(model_path)

        backend = _BackendConfig(C.sizeof(_BackendConfig), gpu)
        model = _ModelConfig(C.sizeof(_ModelConfig), self.model_path.encode(), None)
        # CTC 필드는 RNNT 모델(Nemotron)에서는 무시된다. 값은 공식 예제와 동일.
        streaming = _StreamingConfig(C.sizeof(_StreamingConfig), 0.16, 1.92, 1.92, right_context)
        cfg = _RecognizerConfig(size=C.sizeof(_RecognizerConfig))
        cfg.backend = C.pointer(backend)
        cfg.model = C.pointer(model)
        cfg.streaming = C.pointer(streaming)

        handle = _vp()
        self._check(self._lib.nemo_speech_asr_create(C.byref(cfg), C.byref(handle)), "모델 로드")
        self._handle = handle

    @property
    def version(self) -> str:
        return (self._lib.nemo_speech_asr_version() or b"").decode()

    def _check(self, status: int, what: str) -> None:
        if status != _OK:
            err = (self._lib.nemo_speech_asr_last_error() or b"").decode(errors="replace")
            raise NemoASRError(f"{what} 실패 (status={status}): {err}")

    def _options(self, language: str | None, interim: bool, contexts: Sequence[SpeechContext]) -> _RecognitionOptions:
        """반환된 opts 는 문자열/배열 버퍼를 _objects 로 참조하므로, 사용하는 동안 살아 있어야 한다."""
        opts = self._lib.nemo_speech_asr_recognition_options_default()
        opts.language_code = language.encode() if language and language != "auto" else None
        opts.interim_results = interim
        opts.enable_automatic_punctuation = True
        contexts = [c for c in contexts if c.phrases]
        if contexts:
            arr = (_SpeechContext * len(contexts))()
            for i, c in enumerate(contexts):
                phrases = (C.c_char_p * len(c.phrases))(*(p.encode() for p in c.phrases))
                arr[i] = _SpeechContext(C.sizeof(_SpeechContext), phrases, len(c.phrases), c.boost)
            opts.speech_contexts = arr
            opts.speech_context_count = len(contexts)
        return opts

    def _read_result(self, res: _vp) -> ASRResult:
        lib = self._lib
        text = ""
        language = None
        if lib.nemo_speech_asr_result_alternative_count(res):
            text = (lib.nemo_speech_asr_result_transcript(res, 0) or b"").decode(errors="replace")
            if lib.nemo_speech_asr_result_language_count(res, 0):
                language = (lib.nemo_speech_asr_result_language_code(res, 0, 0) or b"").decode() or None
        # 자동 감지 모드에서는 "...문장. <ko-KR>" 처럼 언어 태그가 붙을 수 있다.
        if m := _LANG_TAG.search(text):
            language = language or m.group(1)
            text = _LANG_TAG.sub(" ", text)
        return ASRResult(
            text=text.strip(),
            is_final=bool(lib.nemo_speech_asr_result_is_final(res)),
            audio_processed=float(lib.nemo_speech_asr_result_audio_processed(res)),
            language=language,
        )

    def recognize(
        self, samples: np.ndarray, sample_rate: int, language: str | None = None, contexts: Sequence[SpeechContext] = ()
    ) -> ASRResult:
        """전체 오디오(mono float32)를 한 번에 인식한다. 블로킹 호출."""
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        opts = self._options(language, False, contexts)
        res = _vp()
        self._check(
            self._lib.nemo_speech_asr_recognize_f32(
                self._handle, C.byref(opts), audio.ctypes.data_as(_f32p), audio.size, sample_rate, C.byref(res)
            ),
            "인식",
        )
        try:
            return self._read_result(res) if res else ASRResult("", True, audio.size / sample_rate, None)
        finally:
            if res:
                self._lib.nemo_speech_asr_result_destroy(res)

    def open_stream(
        self, language: str | None = None, interim: bool = True, contexts: Sequence[SpeechContext] = ()
    ) -> NemoStream:
        opts = self._options(language, interim, contexts)
        handle = _vp()
        self._check(self._lib.nemo_speech_asr_streaming_recognize(self._handle, C.byref(opts), C.byref(handle)), "스트림 생성")
        return NemoStream(self, handle, opts)

    def close(self) -> None:
        if self._handle:
            self._lib.nemo_speech_asr_destroy(self._handle)
            self._handle = _vp()


class NemoStream:
    """스트리밍 인식 세션. 하나의 스트림은 반드시 한 스레드에서만 다뤄야 한다."""

    def __init__(self, recognizer: NemoRecognizer, handle: _vp, opts: _RecognitionOptions):
        self._r = recognizer
        self._lib = recognizer._lib
        self._handle = handle
        self._opts = opts  # 라이브러리가 speech context 를 복사하지 않을 수 있으므로 스트림 수명 동안 보관

    def push(self, samples: np.ndarray, sample_rate: int) -> None:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        self._r._check(
            self._lib.nemo_speech_asr_stream_push_f32(self._handle, audio.ctypes.data_as(_f32p), audio.size, sample_rate),
            "오디오 입력",
        )

    def drain(self) -> list[ASRResult]:
        """지금 디코딩 가능한 결과를 모두 꺼낸다. (next() 가 실제 디코딩을 수행한다)"""
        results = []
        while True:
            res = _vp()
            self._r._check(self._lib.nemo_speech_asr_stream_next(self._handle, C.byref(res)), "디코딩")
            if not res:
                return results
            try:
                results.append(self._r._read_result(res))
            finally:
                self._lib.nemo_speech_asr_result_destroy(res)

    def force_endpoint(self) -> None:
        self._r._check(self._lib.nemo_speech_asr_stream_force_endpoint(self._handle), "발화 종료")

    def finish(self) -> None:
        self._r._check(self._lib.nemo_speech_asr_stream_finish(self._handle), "스트림 종료")

    def close(self) -> None:
        if self._handle:
            self._lib.nemo_speech_asr_stream_close(self._handle)
            self._handle = _vp()


if __name__ == "__main__":
    # 스모크 테스트: uv run python -m streaming_asr1.nemo_asr test.wav [ko-KR]
    import time

    from .audio import decode_wav
    from .config import get_settings

    s = get_settings()
    samples, rate = decode_wav(Path(sys.argv[1]).read_bytes())
    language = sys.argv[2] if len(sys.argv) > 2 else s.asr_language

    t0 = time.perf_counter()
    rec = NemoRecognizer(s.nemo_lib_dir, s.asr_model_path, s.asr_gpu, s.asr_right_context)
    print(f"[load] {time.perf_counter() - t0:.1f}s  nemo-speech {rec.version}")

    t0 = time.perf_counter()
    r = rec.recognize(samples, rate, language)
    print(f"[offline] {time.perf_counter() - t0:.2f}s  ({r.language}) {r.text}")

    t0 = time.perf_counter()
    stream = rec.open_stream(language)
    step = rate // 10
    for i in range(0, samples.size, step):
        stream.push(samples[i : i + step], rate)
        for res in stream.drain():
            print(f"  [{'final' if res.is_final else 'partial'} @{res.audio_processed:.2f}s] {res.text}")
    stream.finish()
    for res in stream.drain():
        print(f"  [{'final' if res.is_final else 'partial'} @{res.audio_processed:.2f}s] {res.text}")
    stream.close()
    print(f"[streaming] {time.perf_counter() - t0:.2f}s for {samples.size / rate:.1f}s audio")
    rec.close()
