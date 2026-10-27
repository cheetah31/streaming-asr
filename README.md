# streaming-asr1

마이크 음성을 실시간으로 받아 텍스트로 변환하는 FastAPI STT 서버.
`nvidia/nemotron-3.5-asr-streaming-0.6b` GGUF 모델을 **FastAPI 프로세스 안에서 직접** 로드해 서빙한다.

## 구조

```
브라우저(마이크) ──WS /ws/transcribe (16kHz PCM)──▶ FastAPI ──ctypes──▶ NeMo-Speech.cpp (CUDA/Metal/Vulkan/CPU)
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
| `src/streaming_asr1/setup_runtime.py` | `streaming-asr1-setup`: 플랫폼별 런타임 / 모델 설치, `.env` 생성 |
| `src/streaming_asr1/logging_setup.py` | 콘솔 + 날짜별 파일 로깅 |
| `src/streaming_asr1/speech_context.py` | speech context 파일 로드 / 자동 재로드 |
| `speech_context.yaml` | word boosting 단어 목록 |
| `src/streaming_asr1/static/index.html` | 마이크 녹음 웹 클라이언트 |

## 준비 (Windows / macOS / Linux)

[uv](https://docs.astral.sh/uv/) 만 있으면 된다. Python 3.12 도 uv 가 알아서 받는다.

**1. uv 설치** (한 번만)

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 터미널을 새로 열고 `uv --version` 으로 확인한다.

**2. 의존성 + 런타임 + 모델** (세 OS 모두 같은 명령, 프로젝트 폴더에서)

```
uv sync                         # ngrok 까지 쓰려면: uv sync --extra colab
uv run streaming-asr1-setup     # 런타임 + 모델(약 700MB) + .env
```

`streaming-asr1-setup` 은 OS / CPU / GPU 를 보고 NeMo-Speech.cpp v0.2.0 릴리스에서 맞는 빌드를 골라
`vendor/nemo-speech/<os>-<arch>-<backend>/` 에 받고(SHA-256 확인), 모델을 `models/` 에 받은 뒤 `.env` 의
`NEMO_LIB_DIR` / `ASR_MODEL_PATH` 등을 채운다. 이미 받은 것은 건너뛰고, `.env` 의 다른 값은 건드리지 않는다.

| 플랫폼 | 자동 선택 | `--backend` 로 고를 수 있는 것 |
|---|---|---|
| Windows x86_64 | NVIDIA 드라이버 580+ → `cuda`, 그보다 낮으면 `vulkan`, GPU 없음 → `cpu` | `cuda` `vulkan` `cpu` |
| Linux x86_64 (Colab, WSL 포함) | NVIDIA GPU → `cuda` (CUDA 12 런타임 포함, Turing/SM75 이상), 없으면 `cpu` | `cuda` `vulkan` `cpu` |
| Linux aarch64 (Jetson 등) | 드라이버 580+ → `cuda13`, 그 외 NVIDIA → `cuda12`, 없으면 `cpu` | `cuda13` `cuda12` `vulkan` `cpu` |
| macOS Apple Silicon | `metal` | `metal` `cpu` |
| macOS Intel | `cpu` | `cpu` |

```
uv run streaming-asr1-setup --backend cpu    # 백엔드 지정 (cpu 면 .env 에 ASR_GPU=-1)
uv run streaming-asr1-setup --force          # 런타임 다시 받기
uv run streaming-asr1-setup --skip-model     # 모델은 직접 받은 것을 쓸 때 (ASR_MODEL_PATH 지정)
```

- AMD / Intel GPU 는 `--backend vulkan` 으로 쓸 수 있다 (자동 선택은 NVIDIA 만 판별한다).
- 저장소에 들어 있는 `vendor/nemo-speech-cuda/` (Windows CUDA 빌드)도 그대로 쓸 수 있다. `NEMO_LIB_DIR` 이 없고
  `vendor/nemo-speech/` 에 설치된 런타임도 없으면 Windows 에서는 이 폴더를 쓴다.

## 실행

세 OS 모두 같은 명령이다. 프로젝트 폴더에서 실행한다.

```
uv run streaming-asr1                             # 서버 시작 → http://127.0.0.1:8100
```

브라우저에서 `http://127.0.0.1:8100` → **녹음 시작**. 종료는 `Ctrl+C`.

| 명령 | 용도 |
|---|---|
| `uv run streaming-asr1` | 이 PC 에서만 접속 (`127.0.0.1:8100`) |
| `uv run streaming-asr1 --host 0.0.0.0` | 같은 네트워크의 다른 PC 에서 접속 (`http://<이 PC IP>:8100`) |
| `uv run streaming-asr1 --port 9000` | 포트 변경 |
| `uv run streaming-asr1 --reload` | 코드 수정 시 자동 재시작 (개발용) |
| `uv run streaming-asr1 --ngrok` | ngrok HTTPS 주소로 외부 공개 (아래 참고) |
| `uv run streaming-asr1 --help` | 옵션 보기 |

다른 PC 의 브라우저에서 마이크를 쓰려면 HTTPS 가 필요하다 (브라우저 정책). `--host 0.0.0.0` 으로는 파일 API 만 쓰고,
원격 마이크 테스트는 `--ngrok` 을 쓴다.

### 스모크 테스트

서버 없이 모델 로드와 인식만 확인한다 (16kHz mono WAV 권장).

```
uv run python -m streaming_asr1.nemo_asr test.wav ko-KR
```

서버가 떠 있으면:

```
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8100/v1/audio/transcriptions -F file=@test.wav -F language=ko
```

Windows PowerShell 5.1 에서는 `curl` 이 `Invoke-WebRequest` 별칭이므로 `curl.exe` 로 쓴다.

### ngrok 으로 외부 공개 (로컬)

[ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) 을 받아 `.env` 에 `NGROK_AUTHTOKEN=...` 을 넣거나 환경변수로 준다.

```powershell
# Windows (PowerShell)
uv sync --extra colab
$env:NGROK_AUTHTOKEN = "<토큰>"
uv run streaming-asr1 --ngrok
```
```bash
# macOS / Linux
uv sync --extra colab
NGROK_AUTHTOKEN=<토큰> uv run streaming-asr1 --ngrok
```

시작 로그에 `ngrok public URL: https://....ngrok-free.app` 이 출력된다. 접속 방법은 아래 Colab 섹션의 **외부 접속** 과 같다.
ngrok 무료 플랜은 터널을 동시에 하나만 열 수 있다.

## Colab GPU + ngrok (`env/colab` 브랜치)

Colab GPU(T4 등)에서 서버를 띄우고 ngrok HTTPS 주소로 외부에서 테스트한다.
노트북 [`colab/streaming_asr1_colab.ipynb`](colab/streaming_asr1_colab.ipynb) 를 Colab 에서 열고 위에서부터 실행하면 된다.

1. 런타임 유형 → **GPU**
2. Colab 보안 비밀에 `NGROK_AUTHTOKEN` 등록 (선택: `NGROK_DOMAIN` 고정 도메인, `GITHUB_TOKEN` 비공개 저장소 clone)
3. 노트북의 `REPO_URL` 을 이 저장소 주소로 바꾼다

노트북이 하는 일:
- `scripts/colab_setup.sh` — uv 설치, `uv sync --extra colab`(ngrok SDK), `uv run streaming-asr1-setup`(linux-x86_64-cuda 런타임 + 모델 + `.env`)
- `uv run streaming-asr1 --ngrok` — 서버와 ngrok 터널을 같은 프로세스에서 띄우고 `ngrok public URL: https://...` 를 출력한다

외부 접속:
- 브라우저: `https://<ngrok 주소>` → ngrok 안내 페이지에서 **Visit Site** → 녹음 시작 (HTTPS 라 원격 마이크 사용 가능)
- API: ngrok 무료 플랜 안내 페이지를 건너뛰도록 헤더를 붙인다
  ```
  curl -H "ngrok-skip-browser-warning: 1" https://<ngrok 주소>/v1/audio/transcriptions -F file=@test.wav -F language=ko
  ```
- WebSocket: `wss://<ngrok 주소>/ws/transcribe?language=ko-KR`

노트북 없이 Colab 터미널(또는 셀)에서 직접 실행할 때:

```bash
git clone -b env/colab <저장소 주소> streaming-asr1 && cd streaming-asr1
bash scripts/colab_setup.sh
NGROK_AUTHTOKEN=<토큰> uv run streaming-asr1 --ngrok
```

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
| `NEMO_LIB_DIR` | `vendor/nemo-speech/` 에 설치된 런타임 (`streaming-asr1-setup` 이 설정) | NeMo-Speech 런타임 라이브러리 폴더 (Windows `bin/`, macOS·Linux `lib/`) |
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
| `NGROK_AUTHTOKEN` | | `--ngrok` 사용 시 필수 |
| `NGROK_DOMAIN` | | ngrok 고정 도메인. 없으면 실행마다 임의 주소 |
