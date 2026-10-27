import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import decode_wav, float_to_pcm16, resample
from .config import get_settings, normalize_language
from .nemo_asr import NemoASRError, NemoRecognizer, SpeechContext
from .session import StreamingTranscriber
from .speech_context import SpeechContextStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    log.info("loading %s (gpu=%d, right_context=%d)", s.asr_model_path, s.asr_gpu, s.asr_right_context)
    t0 = time.perf_counter()
    recognizer = await asyncio.to_thread(NemoRecognizer, s.nemo_lib_dir, s.asr_model_path, s.asr_gpu, s.asr_right_context)
    log.info("model loaded in %.1fs (nemo-speech %s)", time.perf_counter() - t0, recognizer.version)
    app.state.recognizer = recognizer
    app.state.speech_context = SpeechContextStore(s.speech_context_path)
    yield
    recognizer.close()


# ---- Swagger 문서 ----

TAG_FILE = "파일 전사"
TAG_TEST = "스트리밍 테스트"
TAG_CONTEXT = "Speech Context"
TAG_SYSTEM = "시스템"

DESCRIPTION = """
`nvidia/nemotron-3.5-asr-streaming-0.6b` 모델을 NeMo-Speech.cpp(CUDA)로 서버 프로세스 안에서 직접 서빙하는 STT 서버.

### 실시간 STT — `WS /ws/transcribe` (Swagger 에는 표시되지 않음)

웹 클라이언트: [`/`](/) (마이크 녹음)

쿼리: `language` (기본 `.env` 의 `ASR_LANGUAGE`), `speech_context` (기본 `true`)

| 방향 | 메시지 |
|---|---|
| 클라이언트 → 서버 | binary: 16kHz mono int16 little-endian PCM (100ms 단위 권장) |
| 클라이언트 → 서버 | text: `{"type":"stop"}` — 진행 중인 발화를 즉시 확정 |
| 서버 → 클라이언트 | `{"type":"ready","sample_rate":16000,"language":"ko-KR"}` |
| | `{"type":"speech_start","id":0,"audio_start":2.4}` |
| | `{"type":"partial","id":0,"text":"안녕하세","stt_ms":6.8}` |
| | `{"type":"final","id":0,"text":"안녕하세요.","language":"ko-KR","audio_end":3.1,"stt_ms":5.1}` |
| | `{"type":"discard","id":1}` — 음성으로 감지됐지만 인식된 텍스트가 없음 |
| | `{"type":"error","id":2,"message":"..."}` |

### Speech context (word boosting)

`speech_context.yaml` 의 단어/구문을 인식 시 우선시한다. 파일을 저장하면 다음 요청부터 자동 반영된다.
현재 적용 내용은 `GET /api/speech-context` 로 확인한다.
"""

OPENAPI_TAGS = [
    {"name": TAG_FILE, "description": "WAV 파일 한 개를 통째로 인식한다 (비실시간, 오프라인 디코딩)."},
    {
        "name": TAG_TEST,
        "description": "녹음 파일을 실시간 마이크와 **같은 스트리밍 경로**(VAD → partial/final)로 흘려보내 "
        "발화 분리 결과와 STT 처리 시간을 측정한다.",
    },
    {"name": TAG_CONTEXT, "description": "word boosting 단어 목록(`speech_context.yaml`) 조회."},
    {"name": TAG_SYSTEM, "description": "서버 / 모델 상태."},
]

app = FastAPI(
    title="Streaming ASR (Nemotron 3.5 ASR / NeMo-Speech.cpp)",
    description=DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---- 응답 모델 ----


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(description="로드된 GGUF 모델 파일명")
    runtime: str = Field(description="NeMo-Speech 런타임 버전")
    device: str = Field(description="`gpu:N` 또는 `cpu`", examples=["gpu:0"])
    default_language: str = Field(examples=["ko-KR"])
    speech_context_phrases: int = Field(description="현재 적용 중인 speech context 구문 수")


class TranscribeResponse(BaseModel):
    text: str = Field(description="인식된 텍스트", examples=["안녕하세요. 음성인식 테스트입니다."])
    language: str | None = Field(description="인식 언어 (자동 감지 시 감지된 언어)", examples=["ko-KR"])
    duration: float = Field(description="오디오 길이(초)", examples=[7.0])


class MsStats(BaseModel):
    avg: float
    p50: float
    p90: float
    max: float


class SttStats(BaseModel):
    partial: MsStats | None = Field(description="partial 결과마다의 처리 시간")
    first_partial: MsStats | None = Field(description="발화별 첫 partial 의 처리 시간")
    final: MsStats | None = Field(description="final 결과의 처리 시간")


class Utterance(BaseModel):
    id: int
    start: float | None = Field(description="발화 시작 시각(초, VAD 기준)")
    end: float | None = Field(description="발화 끝 시각(초, 모델 기준)")
    text: str
    partials: int = Field(description="이 발화에서 나온 partial 수")
    language: str | None = None
    first_partial_stt_ms: float | None = None
    final_stt_ms: float | None = None
    error: str | None = None


class ConversationResponse(BaseModel):
    file: str | None
    language: str
    speech_context: bool = Field(description="speech context 적용 여부")
    duration: float = Field(description="오디오 길이(초)")
    elapsed: float = Field(description="전체 처리 시간(초)")
    rtf: float | None = Field(description="처리 시간 / 오디오 길이 (1 미만이면 실시간보다 빠름)")
    text: str = Field(description="모든 발화를 이어 붙인 텍스트")
    utterances: list[Utterance]
    discarded: int = Field(description="음성으로 감지됐지만 텍스트가 없는 구간 수")
    stt_ms: SttStats = Field(description="STT 처리 시간 통계 (ms, 100ms 오디오 청크 기준)")


class ContextGroupInfo(BaseModel):
    name: str
    boost: float
    phrases: list[str]


class SpeechContextResponse(BaseModel):
    path: str = Field(description="speech context 파일 경로")
    exists: bool = Field(description="파일 존재 여부 (없으면 컨텍스트 없이 인식)")
    loaded_at: str | None = Field(description="마지막으로 정상 로드된 시각")
    error: str | None = Field(description="마지막 로드 실패 사유. 실패 시 이전에 로드된 내용을 계속 사용한다")
    group_count: int
    phrase_count: int
    groups: list[ContextGroupInfo]


# ---- 공통 ----

LanguageForm = Form(None, description="`ko`, `ko-KR`, `en` … 또는 `auto`(자동 감지). 비우면 서버 기본값")
SpeechContextForm = Form(True, description="`speech_context.yaml` 의 word boosting 적용 여부 (끄고 비교할 때 `false`)")
WavFile = File(..., description="PCM WAV 파일 (8~96kHz, mono/stereo, 8/16/24/32bit)")


def _contexts(enabled: bool) -> list[SpeechContext]:
    return app.state.speech_context.contexts() if enabled else []


async def _transcribe_upload(file: UploadFile, language: str | None, speech_context: bool) -> dict:
    s = get_settings()
    try:
        samples, rate = decode_wav(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        result = await asyncio.to_thread(
            app.state.recognizer.recognize,
            samples,
            rate,
            normalize_language(language, s.asr_language),
            _contexts(speech_context),
        )
    except NemoASRError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"text": result.text, "language": result.language, "duration": round(samples.size / rate, 2)}


def _ms_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    return {
        "avg": round(sum(v) / len(v), 1),
        "p50": v[len(v) // 2],
        "p90": v[min(len(v) - 1, int(len(v) * 0.9))],
        "max": v[-1],
    }


# ---- 엔드포인트 ----


@app.get("/", include_in_schema=False)
async def index():
    # 수정된 페이지가 바로 반영되도록 브라우저 캐시를 쓰지 않는다
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health", tags=[TAG_SYSTEM], summary="서버 상태", response_model=HealthResponse)
async def health():
    """로드된 모델, 런타임 버전, 사용 장치, 기본 언어, 적용 중인 speech context 구문 수를 돌려준다."""
    s = get_settings()
    r: NemoRecognizer = app.state.recognizer
    return {
        "status": "ok",
        "model": s.asr_model_path.name,
        "runtime": f"nemo-speech {r.version}",
        "device": "cpu" if s.asr_gpu < 0 else f"gpu:{s.asr_gpu}",
        "default_language": s.asr_language,
        "speech_context_phrases": sum(len(c.phrases) for c in app.state.speech_context.contexts()),
    }


@app.post(
    "/api/transcribe",
    tags=[TAG_FILE],
    summary="WAV 파일 전사",
    response_model=TranscribeResponse,
    responses={400: {"description": "WAV 가 아니거나 지원하지 않는 형식"}, 500: {"description": "인식 실패"}},
)
async def transcribe_file(
    file: UploadFile = WavFile,
    language: str | None = LanguageForm,
    speech_context: bool = SpeechContextForm,
):
    """WAV 파일 한 개를 통째로 인식해 텍스트를 돌려준다.

    짧은 녹음(수십 초 이내)에 적합하다. 긴 대화 녹음은 발화 단위로 나눠 주는
    `POST /api/transcribe/conversation` 을 사용한다.
    """
    return await _transcribe_upload(file, language, speech_context)


@app.post(
    "/v1/audio/transcriptions",
    tags=[TAG_FILE],
    summary="WAV 파일 전사 (OpenAI 호환)",
    responses={
        200: {
            "description": "`response_format` 에 따라 `{\"text\"}`, 평문, 또는 `verbose_json`",
            "content": {"application/json": {}, "text/plain": {}},
        },
        400: {"description": "WAV 가 아니거나 지원하지 않는 형식"},
    },
)
async def openai_transcriptions(
    file: UploadFile = WavFile,
    model: str | None = Form(None, description="OpenAI 호환용. 무시되며 로드된 모델 하나만 사용한다"),
    language: str | None = LanguageForm,
    response_format: Literal["json", "text", "verbose_json"] = Form("json", description="응답 형식"),
):
    """OpenAI `audio/transcriptions` API 와 같은 형식으로 요청/응답한다. (OpenAI SDK 의 `base_url` 만 바꿔 사용 가능)

    speech context 는 항상 적용된다.
    """
    out = await _transcribe_upload(file, language, True)
    if response_format == "text":
        return PlainTextResponse(out["text"])
    if response_format == "verbose_json":
        return {**out, "task": "transcribe"}
    return {"text": out["text"]}


@app.post(
    "/api/transcribe/conversation",
    tags=[TAG_TEST],
    summary="대화 녹음으로 실시간 STT 테스트",
    response_model=ConversationResponse,
    response_model_exclude_none=True,
    responses={400: {"description": "WAV 가 아니거나 speed 가 음수"}, 500: {"description": "스트림 생성 실패"}},
)
async def transcribe_conversation(
    file: UploadFile = WavFile,
    language: str | None = LanguageForm,
    speed: float = Form(0.0, description="0 = 최대 속도, 1 = 실시간(실제 마이크와 같은 타이밍), 2 = 2배속 …"),
    speech_context: bool = SpeechContextForm,
):
    """대화 녹음(WAV)을 실시간 마이크와 같은 스트리밍 경로(VAD + partial/final)로 흘려보내 테스트한다.

    파일을 100ms 단위로 잘라 WebSocket 세션과 동일하게 처리하므로, `.env` 의 `VAD_THRESHOLD_DB`,
    `END_SILENCE_MS` 설정이 그대로 적용된다. 발화별 결과와 STT 처리 시간 통계를 돌려준다.

    - `speed=0` 은 GPU 가 계속 바쁘므로 최대 성능에 가까운 처리 시간이 나온다.
    - `speed=1` 은 실제 마이크 사용 시의 처리 시간(GPU 절전 영향 포함)을 재현한다. 응답까지 오디오 길이만큼 걸린다.
    - `speech_context=false` 로 한 번 더 요청하면 word boosting 효과를 비교할 수 있다.
    """
    s = get_settings()
    try:
        samples, rate = decode_wav(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if speed < 0:
        raise HTTPException(status_code=400, detail="speed 는 0 이상이어야 합니다")
    pcm = float_to_pcm16(resample(samples, rate, s.sample_rate))
    duration = samples.size / rate

    events: list[dict] = []

    async def collect(msg: dict) -> None:
        events.append(msg)

    lang = normalize_language(language, s.asr_language)
    session = StreamingTranscriber(app.state.recognizer, s, collect, lang, _contexts(speech_context))
    try:
        await session.start()
    except NemoASRError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    log.info(
        "conversation test: %s (%.1fs, %dHz, speed=%g, speech_context=%s)",
        file.filename, duration, rate, speed, speech_context,
    )
    chunk = s.sample_rate // 10 * 2  # 100ms (int16)
    t0 = time.perf_counter()
    try:
        for i in range(0, len(pcm), chunk):
            session.feed(pcm[i : i + chunk])
            if speed > 0:
                await asyncio.sleep(0.1 / speed)
            else:
                await session.flush()
    finally:
        await session.close()  # 남은 오디오를 디코딩하고 마지막 발화를 확정한다
    elapsed = time.perf_counter() - t0

    # 이벤트를 발화 단위로 묶는다
    utts: dict[int, dict] = {}
    for e in events:
        if "id" not in e:
            continue
        u = utts.setdefault(e["id"], {"id": e["id"], "start": None, "end": None, "text": "", "partials": 0})
        if e["type"] == "speech_start":
            u["start"] = e["audio_start"]
        elif e["type"] == "partial":
            u["partials"] += 1
            u.setdefault("first_partial_stt_ms", e["stt_ms"])
        elif e["type"] == "final":
            u.update(text=e["text"], end=e["audio_end"], language=e["language"], final_stt_ms=e["stt_ms"])
        elif e["type"] == "error":
            u["error"] = e["message"]
    utterances = [u for u in utts.values() if u["text"] or "error" in u]

    return {
        "file": file.filename,
        "language": lang,
        "speech_context": speech_context,
        "duration": round(duration, 2),
        "elapsed": round(elapsed, 2),
        "rtf": round(elapsed / duration, 3) if duration else None,
        "text": " ".join(u["text"] for u in utterances if u["text"]),
        "utterances": utterances,
        "discarded": len(utts) - len(utterances),
        "stt_ms": {
            "partial": _ms_stats([e["stt_ms"] for e in events if e["type"] == "partial"]),
            "first_partial": _ms_stats([u["first_partial_stt_ms"] for u in utts.values() if "first_partial_stt_ms" in u]),
            "final": _ms_stats([e["stt_ms"] for e in events if e["type"] == "final"]),
        },
    }


@app.get(
    "/api/speech-context",
    tags=[TAG_CONTEXT],
    summary="현재 적용된 speech context 조회",
    response_model=SpeechContextResponse,
)
async def get_speech_context():
    """`speech_context.yaml` 에서 읽어 현재 인식에 적용 중인 단어/구문 목록과 boost 값을 돌려준다.

    파일이 바뀌었으면 이 요청 시점에 다시 읽는다. 파일 형식이 잘못되면 `error` 에 사유가 표시되고,
    마지막으로 정상 로드된 내용이 계속 적용된다.

    이미 열려 있는 WebSocket 세션은 연결 시점의 컨텍스트를 유지하며, 새 연결부터 반영된다.
    """
    return app.state.speech_context.info()


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket, language: str | None = None, speech_context: bool = True):
    """실시간 전사. 메시지 형식은 앱 description(Swagger 상단) 참고."""
    s = get_settings()
    await ws.accept()
    send_lock = asyncio.Lock()
    closed = False

    async def send(msg: dict) -> None:
        if closed:
            return
        async with send_lock:
            try:
                await ws.send_json(msg)
            except Exception:
                pass  # 클라이언트가 이미 끊긴 경우

    lang = normalize_language(language, s.asr_language)
    contexts = _contexts(speech_context)
    session = StreamingTranscriber(app.state.recognizer, s, send, lang, contexts)
    try:
        await session.start()
    except NemoASRError as e:
        await send({"type": "error", "message": str(e)})
        await ws.close()
        return

    await send({"type": "ready", "sample_rate": s.sample_rate, "language": lang})
    log.info("ws session ready (language=%s, speech_context=%d phrases)", lang, sum(len(c.phrases) for c in contexts))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                session.feed(data)
            elif (text := msg.get("text")) is not None:
                if json.loads(text).get("type") == "stop":
                    session.request_endpoint()
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()  # 남은 오디오를 마저 디코딩하고 최종 결과 전송 시도
        closed = True
        log.info("ws session closed")
