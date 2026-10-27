import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

LOG_PREFIX = "streaming-asr1"


class DailyFileHandler(logging.FileHandler):
    """날짜별 파일(logs/streaming-asr1-YYYY-MM-DD.log)에 기록하고, 날짜가 바뀌면 새 파일로 넘어간다.

    TimedRotatingFileHandler 는 파일 rename 으로 회전하는데, Windows 에서는 다른 프로세스
    (uvicorn --reload 등)가 파일을 열고 있으면 rename 이 실패하므로 처음부터 날짜별 파일에 쓴다.
    retention_days 보다 오래된 파일은 날짜가 바뀔 때 삭제한다.
    """

    def __init__(self, log_dir: str | os.PathLike, retention_days: int = 30):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._day = date.today()
        super().__init__(self._path(self._day), encoding="utf-8", delay=True)
        self._cleanup()

    def _path(self, day: date) -> Path:
        return self._dir / f"{LOG_PREFIX}-{day:%Y-%m-%d}.log"

    def _cleanup(self) -> None:
        cutoff = f"{LOG_PREFIX}-{self._day - timedelta(days=self._retention_days):%Y-%m-%d}.log"
        for f in self._dir.glob(f"{LOG_PREFIX}-????-??-??.log"):
            if f.name < cutoff:
                f.unlink(missing_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        day = datetime.fromtimestamp(record.created).date()
        if day != self._day:
            # handle() 가 이미 lock 을 잡고 있다. 스트림만 닫으면 다음 emit 에서 새 파일을 연다.
            if self.stream:
                self.stream.close()
                self.stream = None
            self._day = day
            self.baseFilename = os.path.abspath(self._path(day))
            self._cleanup()
        super().emit(record)


def logging_config(log_dir: Path, retention_days: int) -> dict:
    """uvicorn 의 log_config 로 넘긴다. 앱/uvicorn 로그를 모두 콘솔과 날짜별 파일에 기록한다."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
            "file": {
                "()": DailyFileHandler,
                "log_dir": str(log_dir),
                "retention_days": retention_days,
                "formatter": "default",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
        "loggers": {
            "uvicorn": {"level": "INFO", "handlers": [], "propagate": True},
            "uvicorn.error": {"level": "INFO", "handlers": [], "propagate": True},
            "uvicorn.access": {"level": "INFO", "handlers": [], "propagate": True},
        },
    }
