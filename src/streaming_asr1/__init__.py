def main() -> None:
    import argparse

    import uvicorn

    from .config import get_settings
    from .logging_setup import logging_config

    parser = argparse.ArgumentParser(description="Nemotron ASR(NeMo-Speech.cpp) 기반 실시간 STT 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--ngrok", action="store_true", help="ngrok 터널로 외부에 공개 (NGROK_AUTHTOKEN 필요)")
    args = parser.parse_args()

    s = get_settings()
    listener = None  # 터널은 이 객체가 살아 있는 동안 유지된다
    if args.ngrok:
        if not s.ngrok_authtoken:
            parser.error("--ngrok 을 쓰려면 NGROK_AUTHTOKEN 을 환경변수나 .env 에 설정하세요")
        import ngrok  # uv sync --extra colab

        options = {"authtoken": s.ngrok_authtoken}
        if s.ngrok_domain:
            options["domain"] = s.ngrok_domain
        listener = ngrok.forward(f"127.0.0.1:{args.port}", **options)
        print(f"ngrok public URL: {listener.url()}", flush=True)

    uvicorn.run(
        "streaming_asr1.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=logging_config(s.log_dir, s.log_retention_days),
    )
