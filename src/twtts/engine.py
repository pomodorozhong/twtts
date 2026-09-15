from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
PRIMETTS_ROOT = Path(os.environ.get("TWTTS_PRIMETTS_DIR", ROOT / "models" / "PrimeTTS"))
PRIMETTS_FILE = PRIMETTS_ROOT / "v21_mbistft_16k" / "primetts_v21_3voice.onnx"
PRIMETTS_SCRIPTS = PRIMETTS_ROOT / "scripts"
BREEZE2_ROOT = Path(
    os.environ.get("TWTTS_BREEZE2_DIR", ROOT / "models" / "Breeze2-VITS-onnx")
)

# Keep all runtime data project-local, including when invoked as plain `uv run`.
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("NLTK_DATA", str(ROOT / ".cache" / "nltk_data"))
os.environ.setdefault("ORT_DISABLE_TELEMETRY_EVENTS", "1")

VOICE_IDS = {
    "xinran": 0,
    "female": 0,
    "anchen": 1,
    "male1": 1,
    "bowen": 2,
    "male2": 2,
}


class PrimeTTSEngine:
    """PrimeTTS v2.1 ONNX inference with a Taiwan Bopomofo frontend."""

    def __init__(self, threads: int | None = None) -> None:
        self.sample_rate = 16_000
        if not PRIMETTS_FILE.exists():
            raise RuntimeError("PrimeTTS is not installed. Run `uv run twtts-setup` first.")
        if str(PRIMETTS_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(PRIMETTS_SCRIPTS))
        import frontend_bopomofo  # type: ignore

        # g2pw defaults to worker processes, which adds latency and can fail in
        # sandboxed/macOS service contexts. Keep pronunciation inference in-process.
        if frontend_bopomofo._g2pw is None:
            g2pw_dir = ROOT / ".cache" / "G2PWModel"
            frontend_bopomofo._g2pw = frontend_bopomofo.G2PWConverter(model_dir=str(g2pw_dir))
            frontend_bopomofo._g2pw.num_workers = 0
            frontend_bopomofo._g2pen = frontend_bopomofo.G2p()
        self.frontend = frontend_bopomofo
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(PRIMETTS_FILE), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()

    @staticmethod
    def voice_id(voice: str | int) -> int:
        if isinstance(voice, int) or str(voice).isdigit():
            result = int(voice)
        else:
            result = VOICE_IDS.get(str(voice).lower(), -1)
        if result not in (0, 1, 2):
            raise ValueError("voice must be xinran, anchen, bowen, or 0-2")
        return result

    def synthesize(self, text: str, voice: str | int = "xinran", speed: float = 1.0) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")

        data = self.frontend.text_to_ids(text)
        if not data["phone_ids"]:
            raise ValueError("text produced no pronounceable symbols")

        def add_blanks(values: list[int]) -> np.ndarray:
            return np.array([[0] + [v for item in values for v in (item, 0)]], dtype=np.int64)

        inputs = {
            "x": add_blanks(data["phone_ids"]),
            "tone": add_blanks(data["tone_ids"]),
            "lang": add_blanks(data["lang_ids"]),
            "x_lengths": np.array([2 * len(data["phone_ids"]) + 1], dtype=np.int64),
            "sid": np.array([self.voice_id(voice)], dtype=np.int64),
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0 / speed], dtype=np.float32),
        }
        with self._lock:
            waveform = self.session.run(None, inputs)[0].reshape(-1)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32)


class Breeze2Engine:
    """Breeze2-VITS-ONNX inference through sherpa-onnx."""

    def __init__(self, threads: int | None = None) -> None:
        files = {
            "model": BREEZE2_ROOT / "breeze2-vits.onnx",
            "lexicon": BREEZE2_ROOT / "lexicon.txt",
            "tokens": BREEZE2_ROOT / "tokens.txt",
        }
        if any(not path.exists() for path in files.values()):
            raise RuntimeError("Breeze2-VITS-ONNX is not installed. Run `uv run twtts-setup` first.")

        import sherpa_onnx

        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(files["model"]),
            lexicon=str(files["lexicon"]),
            tokens=str(files["tokens"]),
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            num_threads=threads or max(1, min(4, os.cpu_count() or 1)),
            debug=False,
            provider="cpu",
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model, rule_fsts="", rule_fars="", max_num_sentences=5
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        self.sample_rate = self.tts.sample_rate
        self._lock = threading.Lock()

    def synthesize(self, text: str, voice: str | int = "default", speed: float = 1.0) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if str(voice).lower() not in ("default", "0"):
            raise ValueError("Breeze2 voice must be default or 0")
        with self._lock:
            audio = self.tts.generate(text=text, sid=0, speed=speed)
        if len(audio.samples) == 0:
            raise ValueError("text produced no pronounceable symbols")
        return np.clip(np.asarray(audio.samples), -1.0, 1.0).astype(np.float32)


MODELS = {
    "primetts": PrimeTTSEngine,
    "breeze2": Breeze2Engine,
}


class TTSEngine:
    """Common interface for the installed Taiwanese Mandarin TTS models."""

    def __init__(self, model: str = "primetts", threads: int | None = None) -> None:
        model = model.lower()
        if model not in MODELS:
            raise ValueError(f"model must be one of: {', '.join(MODELS)}")
        self.model = model
        self.backend = MODELS[model](threads=threads)
        self.sample_rate = self.backend.sample_rate

    def synthesize(
        self, text: str, voice: str | int | None = None, speed: float = 1.0
    ) -> np.ndarray:
        default_voice = "xinran" if self.model == "primetts" else "default"
        return self.backend.synthesize(text, voice or default_voice, speed)

    def encode(self, waveform: np.ndarray, format: str = "mp3") -> bytes:
        format = format.lower()
        if format == "wav":
            output = io.BytesIO()
            sf.write(output, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
            return output.getvalue()
        if format != "mp3":
            raise ValueError("format must be mp3 or wav")
        if not shutil_which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP3 output")
        wav = self.encode(waveform, "wav")
        process = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "96k", "pipe:1"],
            input=wav,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", "replace"))
        return process.stdout


def shutil_which(program: str) -> str | None:
    from shutil import which

    return which(program)
