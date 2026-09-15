# twtts

Standalone, fully local Taiwanese Mandarin TTS. It uses **PrimeTTS v2.1**, a conventional
37.9M-parameter MB-iSTFT-VITS acoustic model—not an LLM or macOS system voice.

## Setup

```bash
uv run twtts-setup
```

## Generate audio

```bash
uv run twtts "大家好，這是臺灣華語語音合成測試。" -o output/hello.mp3
uv run twtts "研究品質很重要。" --voice anchen --speed 1.1 -o output/hello.wav
echo "歡迎光臨" | uv run twtts - --format mp3 > output/stdout.mp3
```

Voices: `xinran` (female), `anchen` (male), and `bowen` (male).

## Local HTTP audio endpoint

```bash
uv run twtts-server
curl -G http://127.0.0.1:8000/tts \
  --data-urlencode 'text=您好，歡迎使用語音服務。' \
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
- Runtime: ONNX Runtime on CPU
- Frontend: Taiwan Bopomofo via G2PW, with Traditional Chinese and English code-mixing
- Audio: 16 kHz mono; MP3 encoding uses `ffmpeg`
