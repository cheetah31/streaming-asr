import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import pcm16_to_float, rms_dbfs
from .config import Settings
from .nemo_asr import ASRResult, NemoASRError, NemoRecognizer, NemoStream, SpeechContext

log = logging.getLogger(__name__)

SendFn = Callable[[dict], Awaitable[None]]
_ENDPOINT = "endpoint"


class StreamingTranscriber:
    """마이크 PCM 을 NeMo-Speech 스트리밍 인식기에 그대로 흘려보낸다.

    - 모델은 cache-aware 스트리밍(RNNT)이라 오디오를 계속 push 하면 interim 결과가 나온다 → "partial"
    - 에너지 기반 VAD 로 "음성 후 무음"을 감지하면 force_endpoint 로 발화를 확정한다 → "final"
    - NeMo 스트림은 한 스레드에서만 다뤄야 하므로 전용 단일 스레드 executor 에서 모든 호출을 수행한다.
    """

    def __init__(
        self,
        recognizer: NemoRecognizer,
        settings: Settings,
        send: SendFn,
        language: str,
        contexts: Sequence[SpeechContext] = (),
    ):
        self._recognizer = recognizer
        self._s = settings
        self._send = send
        self._language = language
        self._contexts = contexts

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-stream")
        self._stream: NemoStream | None = None
        self._queue: asyncio.Queue[bytes | str | asyncio.Future | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

        # VAD 상태
        self._in_speech = False
        self._silence_ms = 0.0
        self._utt_ms = 0.0

        # 출력 상태
        self._audio_samples = 0  # 지금까지 받은 오디오 길이 (발화 시작 시각 계산용)
        self._utt_id = 0
        self._last_partial = ""

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def start(self) -> None:
        self._stream = await self._run(self._recognizer.open_stream, self._language, True, self._contexts)
        self._worker = asyncio.create_task(self._process())

    def feed(self, pcm: bytes) -> None:
        """16kHz mono int16 PCM 을 큐에 넣는다. (디코딩은 워커가 비동기로 수행)"""
        self._queue.put_nowait(pcm)

    def request_endpoint(self) -> None:
        """현재 발화를 즉시 확정한다. (녹음 중지 버튼 등)"""
        self._queue.put_nowait(_ENDPOINT)

    async def flush(self) -> None:
        """지금까지 넣은 오디오의 디코딩과 결과 전송이 끝날 때까지 기다린다."""
        fut = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(fut)
        await fut

    async def close(self) -> None:
        if self._worker:
            self._queue.put_nowait(None)
            await self._worker
        if self._stream:
            await self._run(self._stream.close)
        self._executor.shutdown(wait=False)

    # ---- 내부 ----

    def _vad(self, samples: np.ndarray) -> tuple[bool, bool]:
        """(speech_started, should_endpoint) 를 반환한다."""
        chunk_ms = samples.size * 1000.0 / self._s.sample_rate
        is_voice = rms_dbfs(samples) > self._s.vad_threshold_db
        if not self._in_speech:
            if is_voice:
                self._in_speech, self._silence_ms, self._utt_ms = True, 0.0, chunk_ms
                return True, False
            return False, False
        self._utt_ms += chunk_ms
        self._silence_ms = 0.0 if is_voice else self._silence_ms + chunk_ms
        if self._silence_ms >= self._s.end_silence_ms or self._utt_ms >= self._s.max_utterance_ms:
            self._in_speech = False
            return False, True
        return False, False

    def _execute(self, ops: list[tuple], finish: bool) -> tuple[list[ASRResult], float]:
        """executor 스레드에서 push / force_endpoint / finish 를 순서대로 실행하고 결과와 소요 시간(ms)을 반환한다."""
        assert self._stream is not None
        t0 = time.perf_counter()
        results: list[ASRResult] = []
        for op in ops:
            if op[0] == "push":
                self._stream.push(op[1], self._s.sample_rate)
            else:
                results += self._stream.drain()
                self._stream.force_endpoint()
        if finish:
            self._stream.finish()
        results += self._stream.drain()
        return results, (time.perf_counter() - t0) * 1000

    async def _process(self) -> None:
        done = False
        while not done:
            # 디코딩하는 동안 쌓인 입력을 한 번에 처리한다.
            items = [await self._queue.get()]
            while not self._queue.empty():
                items.append(self._queue.get_nowait())

            ops: list[tuple] = []
            pending: list[np.ndarray] = []
            waiters: list[asyncio.Future] = []
            for item in items:
                if item is None:
                    done = True
                    break
                if isinstance(item, asyncio.Future):
                    waiters.append(item)
                    continue
                if item == _ENDPOINT:
                    if pending:
                        ops.append(("push", np.concatenate(pending)))
                        pending = []
                    ops.append((_ENDPOINT,))
                    self._in_speech = False
                    continue
                samples = pcm16_to_float(item)
                started, endpoint = self._vad(samples)
                if started:
                    audio_start = round(self._audio_samples / self._s.sample_rate, 2)
                    log.info("utterance %d speech start @%.2fs", self._utt_id, audio_start)
                    await self._send({"type": "speech_start", "id": self._utt_id, "audio_start": audio_start})
                self._audio_samples += samples.size
                pending.append(samples)
                if endpoint:
                    ops.append(("push", np.concatenate(pending)))
                    ops.append((_ENDPOINT,))
                    pending = []
            if pending:
                ops.append(("push", np.concatenate(pending)))

            audio_ms = sum(op[1].size for op in ops if op[0] == "push") * 1000 / self._s.sample_rate
            try:
                results, stt_ms = await self._run(self._execute, ops, done)
            except NemoASRError as e:
                log.error("streaming ASR failed: %s", e)
                await self._send({"type": "error", "id": self._utt_id, "message": str(e)})
                results, stt_ms = [], 0.0
            for r in results:
                await self._emit(r, stt_ms, audio_ms)
            for w in waiters:
                w.set_result(None)

    async def _emit(self, r: ASRResult, stt_ms: float, audio_ms: float) -> None:
        """stt_ms: 이 결과를 낸 push+디코딩 호출의 소요 시간, audio_ms: 그 호출에 넣은 오디오 길이"""
        if not r.is_final:
            if r.text and r.text != self._last_partial:
                self._last_partial = r.text
                log.info("utterance %d partial (stt %.1fms, audio %.0fms): %r", self._utt_id, stt_ms, audio_ms, r.text)
                await self._send({"type": "partial", "id": self._utt_id, "text": r.text, "stt_ms": round(stt_ms, 1)})
            return
        log.info("utterance %d final (stt %.1fms, audio %.0fms): %r", self._utt_id, stt_ms, audio_ms, r.text)
        if r.text:
            await self._send({
                "type": "final",
                "id": self._utt_id,
                "text": r.text,
                "language": r.language,
                "audio_end": round(r.audio_processed, 2),
                "stt_ms": round(stt_ms, 1),
            })
        else:
            await self._send({"type": "discard", "id": self._utt_id})
        self._utt_id += 1
        self._last_partial = ""
