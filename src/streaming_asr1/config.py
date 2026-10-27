from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .setup_runtime import default_lib_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """환경변수 또는 .env 파일로 설정한다. (예: ASR_MODEL_PATH=C:/models/xxx.gguf)"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # NeMo-Speech.cpp 런타임 / 모델
    # libnemo_speech_asr_c(.dll/.so/.dylib) 가 있는 폴더. 기본값: streaming-asr1-setup 으로 설치한 런타임
    nemo_lib_dir: Path = Field(default_factory=default_lib_dir)
    asr_model_path: Path  # nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
    asr_gpu: int = 0  # -1 = CPU
    asr_right_context: int = 1  # 0,1,3,6,13 (x80ms 프레임). 클수록 정확하지만 지연이 늘어난다
    asr_language: str = "ko-KR"  # "auto" = 자동 감지
    speech_context_path: Path = Path("speech_context.yaml")  # word boosting 단어 목록. 수정 시 자동 반영

    # 마이크 스트림 / 발화 종료 감지
    sample_rate: int = 16000
    vad_threshold_db: float = -45.0  # 이 레벨(dBFS)보다 크면 음성으로 판단
    end_silence_ms: int = 600  # 음성 뒤 이 시간 이상 조용하면 발화를 확정
    max_utterance_ms: int = 20000  # 발화가 이보다 길면 강제로 확정

    # 로그 (날짜별 파일: logs/streaming-asr1-YYYY-MM-DD.log)
    log_dir: Path = Path("logs")
    log_retention_days: int = 30  # 이보다 오래된 로그 파일은 삭제

    # ngrok 터널 (streaming-asr1 --ngrok)
    ngrok_authtoken: str | None = None  # https://dashboard.ngrok.com/get-started/your-authtoken
    ngrok_domain: str | None = None  # 고정 도메인 (예: xxx.ngrok-free.app). 없으면 실행마다 임의 주소

    @field_validator("nemo_lib_dir", "asr_model_path", "log_dir", "speech_context_path")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        v = v.expanduser()
        return v if v.is_absolute() else PROJECT_ROOT / v


@lru_cache
def get_settings() -> Settings:
    return Settings()


# OpenAI 스타일 2글자 언어 코드 → 모델 locale
_LOCALES = {
    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR", "nl": "nl-NL",
    "de": "de-DE", "tr": "tr-TR", "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP",
    "ko": "ko-KR", "vi": "vi-VN", "uk": "uk-UA", "zh": "zh-CN", "pl": "pl-PL", "sv": "sv-SE",
}


def normalize_language(language: str | None, default: str) -> str:
    language = (language or default).strip()
    return _LOCALES.get(language.lower(), language)
