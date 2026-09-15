from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache"
KOKORO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "kokoro-multi-lang-v1_1.tar.bz2"
)


def download_archive(url: str, destination: Path) -> None:
    """Download and safely extract a sherpa-onnx model archive."""
    destination.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="twtts-", suffix=".tar.bz2", dir=str(CACHE_DIR), delete=False
        ) as temporary:
            archive = Path(temporary.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary.write(chunk)
            except Exception:
                archive.unlink(missing_ok=True)
                raise

    try:
        root = destination.resolve()
        with tarfile.open(archive, mode="r:bz2") as tar:
            members = tar.getmembers()
            for member in members:
                member_path = (destination / member.name).resolve()
                if member_path != root and root not in member_path.parents:
                    raise RuntimeError(f"Refusing archive path outside destination: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Refusing archive link: {member.name}")
            tar.extractall(destination)
    finally:
        archive.unlink(missing_ok=True)


def main() -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
    os.environ.setdefault("NLTK_DATA", str(CACHE_DIR / "nltk_data"))
    os.environ.setdefault("ORT_DISABLE_TELEMETRY_EVENTS", "1")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)

    import nltk
    from huggingface_hub import snapshot_download

    nltk_dir = CACHE_DIR / "nltk_data"
    for package in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"):
        if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
            raise RuntimeError(f"Failed to download NLTK package: {package}")

    print("Downloading PrimeTTS model files…")
    snapshot_download(
        repo_id="Luigi/PrimeTTS",
        local_dir=ROOT / "models" / "PrimeTTS",
        allow_patterns=[
            "v21_mbistft_16k/primetts_v21_3voice.onnx",
            "scripts/frontend_bopomofo.py",
            "scripts/text_norm.py",
            "scripts/symbol_table.json",
        ],
    )

    print("Downloading Breeze2-VITS-ONNX model files…")
    snapshot_download(
        repo_id="MediaTek-Research/Breeze2-VITS-onnx",
        local_dir=ROOT / "models" / "Breeze2-VITS-onnx",
        allow_patterns=["breeze2-vits.onnx", "lexicon.txt", "tokens.txt", "README.md"],
    )

    kokoro_root = ROOT / "models" / "kokoro-multi-lang-v1_1"
    kokoro_files = (
        kokoro_root / "model.onnx",
        kokoro_root / "voices.bin",
        kokoro_root / "tokens.txt",
        kokoro_root / "lexicon-us-en.txt",
        kokoro_root / "lexicon-zh.txt",
    )
    if not all(path.exists() for path in kokoro_files):
        print("Downloading Kokoro multilingual model files…")
        download_archive(KOKORO_URL, ROOT / "models")
    else:
        print("Kokoro multilingual model files already present.")

    # Initialize once so G2PW and its tokenizer are downloaded during setup.
    from .engine import TTSEngine

    print("Initializing the Taiwanese pronunciation model…")
    TTSEngine("primetts")
    print("Checking Breeze2-VITS-ONNX…")
    TTSEngine("breeze2")
    print("Checking Kokoro multilingual…")
    TTSEngine("kokoro")
    print("Setup complete.")
    print("Try: uv run twtts '大家好，這是臺灣華語測試。' -o output/test.mp3")
    print("Or:  uv run twtts --model breeze2 '大家好，這是臺灣華語測試。' -o output/breeze2.mp3")
    print("Or:  uv run twtts --model kokoro 'Hello，這是 Kokoro。' --voice zf_001 -o output/kokoro.mp3")


if __name__ == "__main__":
    main()
