#!/usr/bin/env bash
# Colab(Linux x86_64 + NVIDIA GPU)에서 서버 실행 환경을 준비한다. 여러 번 실행해도 된다.
#   1) uv + Python 의존성(ngrok 포함)
#   2) streaming-asr1-setup: NeMo-Speech.cpp CUDA 런타임, 모델, .env
# 사용: bash scripts/colab_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null; then
  echo "경고: nvidia-smi 가 없습니다. 런타임 유형을 GPU(T4 등)로 바꾸세요." >&2
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra colab
uv run streaming-asr1-setup

echo "실행: NGROK_AUTHTOKEN=... uv run streaming-asr1 --ngrok"
