from __future__ import annotations

import argparse
import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import KOKORO_VOICE_NAMES, TTSEngine

app = FastAPI(title="Local Mandarin TTS", version="0.1.0")
_engines: dict[str, TTSEngine] = {}


def engine(model: str) -> TTSEngine:
    if model not in _engines:
        _engines[model] = TTSEngine(model)
    return _engines[model]


class TTSRequest(BaseModel):
    text: str
    model: str = "primetts"
    voice: str | None = None
    speed: float = 1.0
    format: str = "mp3"


def response_for(request: TTSRequest) -> StreamingResponse:
    try:
        selected = engine(request.model)
        payload = selected.encode(
            selected.synthesize(request.text, request.voice, request.speed), request.format
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
    return StreamingResponse(io.BytesIO(payload), media_type=media_type)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "models": {
            "primetts": {"voices": ["xinran", "anchen", "bowen"]},
            "breeze2": {"voices": ["default"]},
            "kokoro": {"voices": list(KOKORO_VOICE_NAMES), "sample_rate": 24000},
        },
    }


@app.get("/tts")
def tts_get(
    text: str = Query(min_length=1),
    model: str = "primetts",
    voice: str | None = None,
    speed: float = 1.0,
    format: str = "mp3",
) -> StreamingResponse:
    return response_for(TTSRequest(text=text, model=model, voice=voice, speed=speed, format=format))


@app.post("/tts")
def tts_post(request: TTSRequest) -> StreamingResponse:
    return response_for(request)


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Run the local Mandarin TTS API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
