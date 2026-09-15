from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache"


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

    # Initialize once so G2PW and its tokenizer are downloaded during setup.
    from .engine import TTSEngine

    print("Initializing the Taiwanese pronunciation model…")
    TTSEngine()
    print("Setup complete.")
    print("Try: uv run twtts '大家好，這是臺灣華語測試。' -o output/test.mp3")


if __name__ == "__main__":
    main()
