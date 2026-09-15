from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import TTSEngine


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Taiwanese Mandarin TTS (non-LLM models)")
    p.add_argument("text", nargs="?", help="text to speak; reads stdin when omitted")
    p.add_argument("-o", "--output", default="output.mp3", help=".mp3/.wav path, or - for stdout")
    p.add_argument("--format", choices=("mp3", "wav"), help="required only when output is stdout")
    p.add_argument("--model", default="primetts", choices=("primetts", "breeze2"))
    p.add_argument("--voice", help="PrimeTTS: xinran/anchen/bowen; Breeze2: default")
    p.add_argument("--speed", type=float, default=1.0, help="0.5 to 2.0 (default: 1.0)")
    return p


def main() -> None:
    args = parser().parse_args()
    text = args.text if args.text not in (None, "-") else sys.stdin.read()
    suffix = Path(args.output).suffix.lower().lstrip(".") if args.output != "-" else ""
    format = args.format or suffix or "mp3"
    if format not in ("mp3", "wav"):
        parser().error("output must end in .mp3 or .wav")

    engine = TTSEngine(args.model)
    audio = engine.encode(engine.synthesize(text, args.voice, args.speed), format)
    if args.output == "-":
        sys.stdout.buffer.write(audio)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio)
        print(output.resolve())


if __name__ == "__main__":
    main()
