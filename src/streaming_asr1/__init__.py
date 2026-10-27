def main() -> None:
    import argparse

    import uvicorn

    from .config import get_settings
    from .logging_setup import logging_config

    parser = argparse.ArgumentParser(description="Nemotron ASR(NeMo-Speech.cpp) 기반 실시간 STT 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    uvicorn.run(
        "streaming_asr1.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=logging_config(s.log_dir, s.log_retention_days),
    )
