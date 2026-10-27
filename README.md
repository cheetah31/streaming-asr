# streaming-asr1

마이크 음성을 실시간으로 받아 텍스트로 변환하는 FastAPI STT 서버.
`nvidia/nemotron-3.5-asr-streaming-0.6b` GGUF 모델을 **FastAPI 프로세스 안에서 직접** 로드해 서빙한다.

## 구조

```
브라우저(마이크) ──WS /ws/transcribe (16kHz PCM)──▶ FastAPI ──ctypes──▶ NeMo-Speech.cpp (nemo_speech_asr_c.dll, CUDA)
       ◀──────── partial / final JSON ─────────        │                    └─ nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
다른 클라이언트 ──POST /v1/audio/transcriptions (WAV)──▶ │
```

- 모델 런타임은 NVIDIA 공식 [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)의 안정 C ABI(`include/nemo_speech/asr.h`)를 `ctypes`로 호출한다.
- 실시간 경로는 모델의 cache-aware **스트리밍 API**를 그대로 쓴다. 오디오를 계속 push 하면 interim 결과(`partial`)가 나오고,
  서버의 에너지 기반 VAD가 "음성 후 무음"을 감지하면 `force_endpoint`로 발화를 확정(`final`)한다.
- 모델은 서버 시작 시 한 번만 로드되며, WebSocket 연결마다 독립 스트림을 연다.

| 파일 | 역할 |
|---|---|
| `src/streaming_asr1/nemo_asr.py` | NeMo-Speech C ABI ctypes 래퍼 (모델 로드, 오프라인/스트리밍 인식) |
| `src/streaming_asr1/session.py` | WebSocket 스트리밍 세션 (VAD, partial/final) |
| `src/streaming_asr1/app.py` | FastAPI 앱 |
| `src/streaming_asr1/config.py` | 설정 (`.env`) |
| `src/streaming_asr1/logging_setup.py` | 콘솔 + 날짜별 파일 로깅 |
| `src/streaming_asr1/speech_context.py` | speech context 파일 로드 / 자동 재로드 |
| `speech_context.yaml` | word boosting 단어 목록 |
| `src/streaming_asr1/static/index.html` | 마이크 녹음 웹 클라이언트 |

## 준비

1. **NeMo-Speech.cpp 런타임** (Windows CUDA 빌드, v0.2.0)을 `vendor/nemo-speech-cuda/`에 둔다.
   ```
   gh release download v0.2.0 -R NVIDIA/NeMo-Speech.cpp -p "nemo-speech-0.2.0-windows-x86_64-cuda.zip"
   # 압축을 풀어 vendor/nemo-speech-cuda/bin/nemo_speech_asr_c.dll 이 되도록 배치
   ```
   - CUDA 빌드는 CUDA 13 런타임(`cublas64_13.dll`)을 포함하므로 **NVIDIA 드라이버 580 이상**이 필요하다.
     드라이버를 올릴 수 없으면 `-vulkan.zip`(GPU) 또는 `-cpu.zip`을 받고 `NEMO_LIB_DIR`을 그 `bin` 폴더로 바꾼다.
2. **모델**: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`
   ```
   hf download nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
   ```
3. `cp .env.example .env` 후 `ASR_MODEL_PATH` 등을 수정한다.

## 실행

```
# 모델 로드 + 인식 스모크 테스트
uv run python -m streaming_asr1.nemo_asr test.wav ko-KR

# 서버
uv run streaming-asr1                 # http://127.0.0.1:8100
uv run streaming-asr1 --host 0.0.0.0 --port 8100
```

브라우저에서 `http://127.0.0.1:8100` → **녹음 시작**.
(다른 PC에서 접속하면 브라우저 마이크 정책상 HTTPS가 필요하다.)

## API

Swagger: http://127.0.0.1:8100/docs (엔드포인트가 `파일 전사` / `스트리밍 테스트` / `Speech Context` / `시스템` 으로 묶여 있다)

### `WS /ws/transcribe?language=ko-KR&speech_context=true`

- 클라이언트 → 서버: binary = 16kHz mono int16 LE PCM, text = `{"type":"stop"}`(현재 발화 즉시 확정)
- 서버 → 클라이언트:
  ```json
  {"type":"ready","sample_rate":16000,"language":"ko-KR"}
  {"type":"speech_start","id":0,"audio_start":0.1}
  {"type":"partial","id":0,"text":"안녕하세","stt_ms":6.8}
  {"type":"final","id":0,"text":"안녕하세요.","language":"ko-KR","audio_end":1.52,"stt_ms":5.1}
  {"type":"discard","id":1}
  {"type":"error","id":2,"message":"..."}
  ```

### `POST /v1/audio/transcriptions` (OpenAI 호환)

multipart: `file`(PCM WAV, 8–96kHz, mono/stereo), `language`(`ko` 또는 `ko-KR`, `auto`), `response_format`(`json`|`text`|`verbose_json`)

```
curl http://127.0.0.1:8100/v1/audio/transcriptions -F file=@test.wav -F language=ko
```

### `POST /api/transcribe/conversation` (대화 음성 테스트)

대화 녹음(WAV)을 실시간 마이크와 **같은 스트리밍 경로**(VAD → partial/final)로 흘려보내 발화별 결과와 STT 처리 시간 통계를 돌려준다.

multipart: `file`(PCM WAV), `language`, `speed`(`0` = 최대 속도(기본), `1` = 실시간, `2` = 2배속 …),
`speech_context`(`true`(기본) / `false` — 끄고 요청해 word boosting 효과를 비교)

```
curl http://127.0.0.1:8100/api/transcribe/conversation -F file=@meeting.wav -F language=ko
curl http://127.0.0.1:8100/api/transcribe/conversation -F file=@meeting.wav -F speed=1   # 실제 마이크와 같은 타이밍
```
```json
{
  "duration": 32.66, "elapsed": 1.82, "rtf": 0.056,
  "text": "안녕하세요. 회의 시작하겠습니다. 네 지난주 진행 상황부터 ...",
  "utterances": [
    {"id":0,"start":0.1,"end":3.8,"text":"안녕하세요. 회의 시작하겠습니다.","partials":10,"first_partial_stt_ms":7.2,"final_stt_ms":0.8,"language":"ko-KR"}
  ],
  "discarded": 1,
  "stt_ms": {"partial":{"avg":5.5,"p50":5.2,"p90":6.0,"max":12.0}, "first_partial":{...}, "final":{...}}
}
```

### `GET /api/speech-context`

현재 적용 중인 speech context(파일 경로, 로드 시각, 오류, 그룹별 boost / 구문)를 돌려준다.

### `POST /api/transcribe`, `GET /health`

`/api/transcribe` 도 `speech_context` 폼 필드를 받는다.

## Speech context (word boosting)

`speech_context.yaml` 에 적은 단어/구문을 인식 시 우선시한다. 도메인 용어, 고유명사, 자주 틀리는 단어에 쓴다.

```yaml
default_boost: 2.0
contexts:
  - name: 상담 도메인
    boost: 2.0          # 0 초과 5.0 이하. 2.0 부터 0.5 씩 조정
    phrases: [상담사, 음성, 실시간 감지]
```

- 파일을 저장하면 **다음 요청부터 자동 반영**된다 (서버 재시작 불필요). 열려 있는 WebSocket 은 연결 시점 내용을 유지.
- 형식이 잘못되면 이전 내용을 계속 쓰고 `GET /api/speech-context` 의 `error` 와 로그에 사유가 남는다.
- "말하는 대로" 적는다 (`GPU` 보다 `지피유`). boost 가 너무 높으면 비슷한 발음을 끌어당겨 오인식이 늘어난다
  (테스트: boost 3.0 에서 `십 밀리초` → `실시간 감지초`).

## 설정 (`.env`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NEMO_LIB_DIR` | `vendor/nemo-speech-cuda/bin` | NeMo-Speech 런타임 DLL 폴더 |
| `ASR_MODEL_PATH` | (필수) | GGUF 모델 경로 |
| `ASR_GPU` | `0` | GPU 번호, `-1` = CPU |
| `ASR_RIGHT_CONTEXT` | `1` | 0/1/3/6/13 (×80ms). 클수록 정확, 지연 증가 |
| `ASR_LANGUAGE` | `ko-KR` | 기본 인식 언어, `auto` = 자동 감지 |
| `SPEECH_CONTEXT_PATH` | `speech_context.yaml` | word boosting 단어 목록 파일 |
| `VAD_THRESHOLD_DB` | `-45` | 주변 소음이 크면 올리고, 말이 잘리면 내린다 |
| `END_SILENCE_MS` | `600` | 무음이 이만큼 이어지면 발화 확정 |
| `MAX_UTTERANCE_MS` | `20000` | 최대 발화 길이 |
| `LOG_DIR` | `logs` | 로그 폴더. 날짜별 파일 `streaming-asr1-YYYY-MM-DD.log` 로 기록 |
| `LOG_RETENTION_DAYS` | `30` | 이보다 오래된 로그 파일은 삭제 |
