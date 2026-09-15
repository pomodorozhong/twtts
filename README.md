# twtts

Standalone, fully local Mandarin TTS with four conventional acoustic models:
**PrimeTTS v2.1**, **Breeze2-VITS-ONNX**, **Kokoro multilingual v1.1**, and **AISHELL3 VITS**.
Neither runtime is an LLM or a macOS system voice.

## Setup

```bash
hf auth login --force
uv run twtts-setup
```

Hugging Face authentication avoids unauthenticated Hub requests and enables higher rate limits
and faster model downloads.

## Generate audio

```bash
uv run twtts "大家好，這是臺灣華語語音合成測試。" -o output/hello.mp3
uv run twtts --model breeze2 "大家好，這是 Breeze2。" -o output/breeze2.mp3
uv run twtts --model kokoro "Hello, 這是 Kokoro。" --voice zf_001 -o output/kokoro.mp3
uv run twtts --model aishell3 "你好，這是 AISHELL3。" --voice 10 -o output/aishell3.mp3
uv run twtts "研究品質很重要。" --voice anchen --speed 1.1 -o output/hello.wav
echo "歡迎光臨" | uv run twtts - --format mp3 > output/stdout.mp3
```

PrimeTTS voices: `xinran` (female), `anchen` (male), and `bowen` (male). Breeze2 currently
has one voice, selected automatically; use `--voice default` if you want to specify it. Kokoro
supports 103 speakers (`zf_*` Chinese voices, `zm_*` Chinese voices, and `af_maple`, `af_sol`,
and `bf_vale` English voices); pass a name or a numeric ID from 0 to 102.

AISHELL3 has 174 Chinese speakers; pass `default` or a numeric speaker ID from 0 to 173.

## Local HTTP audio endpoint

```bash
uv run twtts-server
curl -G http://127.0.0.1:8000/tts \
  --data-urlencode 'text=您好，歡迎使用語音服務。' \
  --data-urlencode 'model=breeze2' \
  -o output/api.mp3
```

POST also works:

```bash
curl http://127.0.0.1:8000/tts \
  -H 'content-type: application/json' \
  -d '{"text":"現在開始播放。","voice":"xinran","format":"mp3"}' \
  -o output/api.mp3
```

The response is sent as `audio/mpeg` or `audio/wav`. Synthesis is performed locally and the
server binds only to localhost by default. The current model synthesizes a complete utterance
before sending it; stdout and HTTP are streaming transports, not incremental model inference.

## Model and licensing

- Model: [Luigi/PrimeTTS](https://huggingface.co/Luigi/PrimeTTS), Apache-2.0
- Model: [MediaTek Research/Breeze2-VITS-onnx](https://huggingface.co/MediaTek-Research/Breeze2-VITS-onnx); its model card does not currently declare a license
- Model: [Kokoro multilingual v1.1](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh), Apache-2.0
- Model: [AISHELL3 VITS](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html), see the model archive for its license
- Runtime: ONNX Runtime (PrimeTTS) and sherpa-onnx (Breeze2/Kokoro/AISHELL3), on CPU
- Frontend: Taiwan Bopomofo via G2PW for PrimeTTS; sherpa-onnx lexicons for the other models
- Audio: mono, 16 kHz (PrimeTTS), 22.05 kHz (Breeze2), 24 kHz (Kokoro), or 8 kHz (AISHELL3); MP3 encoding uses `ffmpeg`
