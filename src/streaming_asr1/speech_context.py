"""speech context(word boosting) 파일 관리.

YAML 파일에서 인식 시 우선시할 단어/구문 목록을 읽는다. 파일이 바뀌면 다음 요청 때 자동으로 다시 읽으므로
서버를 재시작하지 않아도 된다. 파일 형식은 speech_context.yaml 의 주석 참고.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .nemo_asr import SpeechContext

log = logging.getLogger(__name__)

MAX_BOOST = 5.0  # NeMo-Speech RNNT 기본 상한 (asr.decoder.boosting_max_boost). 넘는 값은 잘린다
DEFAULT_BOOST = 2.5


@dataclass(frozen=True)
class ContextGroup:
    name: str
    boost: float
    phrases: tuple[str, ...]


class SpeechContextError(ValueError):
    pass


def parse(data: object) -> list[ContextGroup]:
    if data is None:
        return []
    if not isinstance(data, dict):
        raise SpeechContextError("최상위는 'contexts:' 를 가진 매핑이어야 합니다")
    default_boost = float(data.get("default_boost", DEFAULT_BOOST))
    groups = []
    for i, item in enumerate(data.get("contexts") or []):
        if not isinstance(item, dict):
            raise SpeechContextError(f"contexts[{i}] 는 매핑이어야 합니다")
        boost = float(item.get("boost", default_boost))
        if not 0 < boost <= MAX_BOOST:
            raise SpeechContextError(f"contexts[{i}].boost 는 0 초과 {MAX_BOOST} 이하여야 합니다: {boost}")
        raw = item.get("phrases") or []
        if not isinstance(raw, list):
            raise SpeechContextError(f"contexts[{i}].phrases 는 목록이어야 합니다")
        phrases = tuple(dict.fromkeys(str(p).strip() for p in raw if str(p).strip()))  # 공백 제거, 중복 제거
        groups.append(ContextGroup(name=str(item.get("name") or f"context-{i}"), boost=boost, phrases=phrases))
    return groups


class SpeechContextStore:
    """speech context 파일을 읽어 두고, 파일 수정 시각이 바뀌면 다시 읽는다.

    파일을 잘못 고쳐 읽기에 실패하면 마지막으로 정상 로드된 내용을 계속 쓰고 오류를 기록한다.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._groups: list[ContextGroup] = []
        self._loaded_at: datetime | None = None
        self._error: str | None = None
        self._refresh()

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        with self._lock:
            if mtime == self._mtime and self._loaded_at is not None:
                return
            self._mtime = mtime
            if mtime is None:
                self._groups, self._error = [], None
                self._loaded_at = datetime.now()
                log.warning("speech context 파일이 없습니다: %s (컨텍스트 없이 인식)", self.path)
                return
            try:
                groups = parse(yaml.safe_load(self.path.read_text(encoding="utf-8")))
            except (OSError, yaml.YAMLError, SpeechContextError, TypeError, ValueError) as e:
                self._error = f"{type(e).__name__}: {e}"
                log.error("speech context 로드 실패, 이전 내용을 유지합니다: %s", self._error)
                return
            self._groups, self._error = groups, None
            self._loaded_at = datetime.now()
            log.info(
                "speech context loaded: %s (%d groups, %d phrases)",
                self.path, len(groups), sum(len(g.phrases) for g in groups),
            )

    def groups(self) -> list[ContextGroup]:
        self._refresh()
        return self._groups

    def contexts(self) -> list[SpeechContext]:
        """인식기에 넘길 형태로 반환한다."""
        return [SpeechContext(g.phrases, g.boost) for g in self.groups() if g.phrases]

    def info(self) -> dict:
        groups = self.groups()
        return {
            "path": str(self.path),
            "exists": self._mtime is not None,
            "loaded_at": self._loaded_at.isoformat(timespec="seconds") if self._loaded_at else None,
            "error": self._error,
            "group_count": len(groups),
            "phrase_count": sum(len(g.phrases) for g in groups),
            "groups": [{"name": g.name, "boost": g.boost, "phrases": list(g.phrases)} for g in groups],
        }
