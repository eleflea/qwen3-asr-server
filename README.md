# Qwen3 ASR Server

A speech recognition service built with FastAPI and mlx-audio.

## Features

- Provides an HTTP API for speech recognition
- Supports optional `language` parameter
- Supports optional `context` prompt (Chinese text is tokenized first to reduce hallucinations)
- Returns forced-alignment timestamps
- Built-in queue-based throttling (default queue size: 10)

## Project Structure

- `asr_server.py`: Main server application
- `serve.sh`: Startup command example
- `requirements.txt`: Dependency list

## Requirements

- Python 3.10+
- macOS (Apple Silicon)
- Access to Hugging Face model hub (or a mirror)

## Install Dependencies

```bash
conda create -n mlx-audio python=3.12 -y
conda activate mlx-audio
pip install -r requirements.txt
```

## Start the Server

```bash
uvicorn asr_server:app --host 127.0.0.1 --port 8000
```

## API Endpoints

### 1) Health Check

- Method: `GET`
- Path: `/health`

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response example:

```json
{
  "status": "ok",
  "queue_length": 0,
  "queue_max_size": 10,
  "model_loaded": true
}
```

### 2) Speech Recognition

- Method: `POST`
- Path: `/asr`
- Form fields:
  - `audio`: audio file (required)
  - `language`: language (optional)
  - `context`: context prompt text (optional)

Example:

```bash
curl -X POST http://127.0.0.1:8000/asr \
  -F "audio=@/path/to/audio.wav" \
  -F "language=English" \
  -F "context=domain-specific prompt text"
```

Response example:

```json
{
  "language": "English",
  "text": "recognized text",
  "timestamps": [
    {
      "start_time": 0.0,
      "end_time": 0.5,
      "text": "recognized"
    }
  ]
}
```

## Notes

- Audio is internally resampled to 16 kHz.
- When the queue is full, the server returns `429 Queue is full`.
- If the audio payload is empty, the server returns `400 Empty audio payload`.
- The first model load can take a while, which is expected.

Current models:

- `mlx-community/Qwen3-ASR-1.7B-8bit`
- `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`

You can find more models [here](https://huggingface.co/mlx-community/models?search=qwen3-asr).
