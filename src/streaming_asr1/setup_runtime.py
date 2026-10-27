"""실행 환경 준비: OS/CPU/GPU 에 맞는 NeMo-Speech.cpp 런타임과 모델을 받고 .env 를 만든다.

    uv run streaming-asr1-setup                   # 자동 선택
    uv run streaming-asr1-setup --backend cpu     # 백엔드 지정

런타임은 vendor/nemo-speech/<os>-<arch>-<backend>/ 에, 모델은 models/ 에 받는다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = PROJECT_ROOT / "vendor" / "nemo-speech"
LEGACY_WINDOWS_LIB_DIR = PROJECT_ROOT / "vendor" / "nemo-speech-cuda" / "bin"  # 저장소에 포함된 Windows CUDA 빌드

NEMO_VERSION = "0.2.0"
RELEASE_URL = "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{version}/{name}"
MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILE = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
MODEL_DIR = PROJECT_ROOT / "models"

# NeMo-Speech.cpp v0.2.0 릴리스가 제공하는 (os, arch) 별 백엔드. 앞쪽일수록 우선
BACKENDS = {
    ("windows", "x86_64"): ["cuda", "vulkan", "cpu"],
    ("linux", "x86_64"): ["cuda", "vulkan", "cpu"],
    ("linux", "aarch64"): ["cuda13", "cuda12", "vulkan", "cpu"],
    ("macos", "aarch64"): ["metal", "cpu"],
    ("macos", "x86_64"): ["cpu"],
}
CUDA13_MIN_DRIVER = 580  # Windows cuda / linux-aarch64 cuda13 빌드 (CUDA 13 런타임)


class SetupError(RuntimeError):
    pass


def current_platform() -> tuple[str, str]:
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    return os_name, arch


def lib_dir(install_dir: Path) -> Path:
    """런타임 폴더에서 libnemo_speech_asr_c 가 있는 하위 폴더 (Windows 는 bin, 나머지는 lib)."""
    return install_dir / ("bin" if sys.platform == "win32" else "lib")


def default_lib_dir() -> Path:
    """설치된 런타임 중 현재 플랫폼에 맞고 우선순위가 가장 높은 것. 없으면 설치될 기본 위치."""
    os_name, arch = current_platform()
    backends = BACKENDS.get((os_name, arch), [])
    for backend in backends:
        d = lib_dir(VENDOR_DIR / f"{os_name}-{arch}-{backend}")
        if d.is_dir():
            return d
    if os_name == "windows" and LEGACY_WINDOWS_LIB_DIR.is_dir():
        return LEGACY_WINDOWS_LIB_DIR
    return lib_dir(VENDOR_DIR / f"{os_name}-{arch}-{backends[-1] if backends else 'cpu'}")


def nvidia_driver_major() -> int | None:
    """NVIDIA 드라이버 메이저 버전. nvidia-smi 가 없으면 None."""
    smi = shutil.which("nvidia-smi") or shutil.which("nvidia-smi", path="/usr/lib/wsl/lib")  # WSL 은 PATH 밖에 있다
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        return int(out.split()[0].split(".")[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def detect_backend(os_name: str, arch: str) -> str:
    if os_name == "macos":
        return "metal" if arch == "aarch64" else "cpu"
    driver = nvidia_driver_major()
    if driver is None:
        return "cpu"
    if os_name == "windows":
        if driver >= CUDA13_MIN_DRIVER:
            return "cuda"
        print(f"NVIDIA 드라이버 {driver} < {CUDA13_MIN_DRIVER}: Windows CUDA 빌드(CUDA 13) 대신 vulkan 을 사용합니다")
        return "vulkan"
    if arch == "aarch64":
        return "cuda13" if driver >= CUDA13_MIN_DRIVER else "cuda12"
    return "cuda"  # linux-x86_64 cuda 빌드는 CUDA 12 런타임 포함


def _download(url: str, dest: Path) -> None:
    headers = {"User-Agent": "streaming-asr1-setup"}
    if "huggingface.co" in url and (token := os.environ.get("HF_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        show = total > 1 << 20
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if show:
                print(f"\r  {dest.name}: {done / total:6.1%} ({done >> 20}/{total >> 20} MB)", end="", flush=True)
    if show:
        print()
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive: Path, dest: Path) -> None:
    """최상위 폴더(nemo-speech-<ver>-<target>/)를 벗겨서 dest 에 푼다."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                rel = Path(*Path(info.filename).parts[1:])
                if info.is_dir() or not rel.parts:
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return
    with tarfile.open(archive) as t:
        members = []
        for m in t.getmembers():
            parts = Path(m.name).parts[1:]
            if parts:
                m.name = "/".join(parts)
                members.append(m)
        t.extractall(dest, members=members, filter="data")  # 심볼릭 링크(libxxx.so -> libxxx.so.1) 유지


def install_runtime(backend: str, version: str = NEMO_VERSION, force: bool = False) -> Path:
    os_name, arch = current_platform()
    available = BACKENDS.get((os_name, arch))
    if not available:
        raise SetupError(f"NeMo-Speech.cpp 릴리스가 {os_name}-{arch} 를 지원하지 않습니다")
    if backend not in available:
        raise SetupError(f"{os_name}-{arch} 에서 쓸 수 있는 백엔드: {', '.join(available)}")

    target = f"{os_name}-{arch}-{backend}"
    install_dir = VENDOR_DIR / target
    if lib_dir(install_dir).is_dir() and not force:
        print(f"런타임 이미 있음: {install_dir.relative_to(PROJECT_ROOT)} (다시 받으려면 --force)")
        return install_dir

    name = f"nemo-speech-{version}-{target}.{'zip' if os_name == 'windows' else 'tar.gz'}"
    print(f"런타임 다운로드: {name}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / name
        _download(RELEASE_URL.format(version=version, name=name), archive)
        _download(RELEASE_URL.format(version=version, name=name + ".sha256"), tmp / "expected.sha256")
        expected = (tmp / "expected.sha256").read_text().split()[0].lower()
        if _sha256(archive) != expected:
            raise SetupError(f"체크섬 불일치: {name}")

        staging = tmp / "extract"
        _extract(archive, staging)
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), install_dir)
    print(f"런타임 설치: {install_dir.relative_to(PROJECT_ROOT)}")
    return install_dir


def download_model() -> Path:
    path = MODEL_DIR / MODEL_FILE
    if path.is_file():
        print(f"모델 이미 있음: {path.relative_to(PROJECT_ROOT)}")
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"모델 다운로드: {MODEL_REPO}/{MODEL_FILE}")
    _download(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}", path)
    return path


def _env_lines() -> list[str]:
    env_path = PROJECT_ROOT / ".env"
    return env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []


def _env_entries(lines: list[str]):
    """(줄 번호, 키, 값) — 주석/빈 줄 제외."""
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            yield i, key.strip(), value.strip()


def update_env(values: dict[str, str], defaults: dict[str, str]) -> None:
    """.env 에서 values 는 덮어쓰고, defaults 는 키가 없을 때만 추가한다. 나머지 줄은 그대로 둔다."""
    env_path = PROJECT_ROOT / ".env"
    lines = _env_lines()
    present = set()
    for i, key, _ in _env_entries(lines):
        present.add(key)
        if key in values:
            lines[i] = f"{key}={values[key]}"
    lines += [f"{k}={v}" for k, v in {**defaults, **values}.items() if k not in present]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f".env 갱신: {', '.join(values)}")


def main() -> None:
    os_name, arch = current_platform()
    parser = argparse.ArgumentParser(description="NeMo-Speech.cpp 런타임 / 모델 설치 및 .env 설정")
    parser.add_argument(
        "--backend", default="auto",
        help=f"auto(기본) 또는 {'/'.join(BACKENDS.get((os_name, arch), ['?']))}  (현재 플랫폼: {os_name}-{arch})",
    )
    parser.add_argument("--version", default=NEMO_VERSION, help="NeMo-Speech.cpp 릴리스 버전")
    parser.add_argument("--force", action="store_true", help="런타임을 다시 받는다")
    parser.add_argument("--skip-model", action="store_true", help="모델 다운로드를 건너뛴다")
    args = parser.parse_args()

    backend = detect_backend(os_name, arch) if args.backend == "auto" else args.backend
    print(f"플랫폼: {os_name}-{arch}, 백엔드: {backend}")
    try:
        install_dir = install_runtime(backend, args.version, args.force)
        model = None if args.skip_model else download_model()
    except (SetupError, OSError) as e:
        sys.exit(f"설치 실패: {e}")

    values = {"NEMO_LIB_DIR": lib_dir(install_dir).relative_to(PROJECT_ROOT).as_posix()}
    current_gpu = next((v for _, k, v in _env_entries(_env_lines()) if k == "ASR_GPU"), None)
    if backend == "cpu":
        values["ASR_GPU"] = "-1"
    elif current_gpu == "-1":  # 이전에 cpu 로 설치했던 경우
        values["ASR_GPU"] = "0"
    defaults = {"ASR_GPU": "0", "ASR_LANGUAGE": "ko-KR"}
    if model:
        defaults["ASR_MODEL_PATH"] = model.relative_to(PROJECT_ROOT).as_posix()
    update_env(values, defaults)
    print("준비 완료. 서버 실행: uv run streaming-asr1")


if __name__ == "__main__":
    main()
