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
MODEL_ROOT = Path(os.environ.get("TWTTS_MODEL_DIR", ROOT / "models" / "PrimeTTS"))
MODEL_FILE = MODEL_ROOT / "v21_mbistft_16k" / "primetts_v21_3voice.onnx"
SCRIPTS_DIR = MODEL_ROOT / "scripts"
SAMPLE_RATE = 16_000

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


class TTSEngine:
    """PrimeTTS v2.1 ONNX inference with a Taiwan Bopomofo frontend."""

    def __init__(self, threads: int | None = None) -> None:
        if not MODEL_FILE.exists():
            raise RuntimeError("PrimeTTS is not installed. Run `uv run twtts-setup` first.")
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
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
            str(MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"]
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

    def encode(self, waveform: np.ndarray, format: str = "mp3") -> bytes:
        format = format.lower()
        if format == "wav":
            output = io.BytesIO()
            sf.write(output, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
