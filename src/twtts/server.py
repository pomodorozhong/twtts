from __future__ import annotations

import argparse
import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import TTSEngine

app = FastAPI(title="Taiwanese Mandarin TTS", version="0.1.0")
_engine: TTSEngine | None = None


def engine() -> TTSEngine:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


class TTSRequest(BaseModel):
    text: str
    voice: str = "xinran"
    speed: float = 1.0
    format: str = "mp3"


def response_for(request: TTSRequest) -> StreamingResponse:
    try:
        payload = engine().encode(
            engine().synthesize(request.text, request.voice, request.speed), request.format
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"
    return StreamingResponse(io.BytesIO(payload), media_type=media_type)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model": "PrimeTTS v2.1", "voices": ["xinran", "anchen", "bowen"]}


@app.get("/tts")
def tts_get(
    text: str = Query(min_length=1), voice: str = "xinran", speed: float = 1.0, format: str = "mp3"
) -> StreamingResponse:
    return response_for(TTSRequest(text=text, voice=voice, speed=speed, format=format))


@app.post("/tts")
def tts_post(request: TTSRequest) -> StreamingResponse:
    return response_for(request)


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Run the local Taiwanese Mandarin TTS API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
