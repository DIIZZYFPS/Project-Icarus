# Project Icarus

A voice-powered AI assistant with real-time speech recognition, natural language processing, and conversational AI capabilities. Named after the mythological figure who dared to fly—though this one knows when to pull back from the sun.

## Architecture

```
┌─────────────────────────────────────┐         ┌──────────────────────────────────────┐
│          CLIENT (Laptop)            │         │           BACKEND (GPU Server)       │
│                                     │         │                                      │
│  PyAudio ──► OpenWakeWord           │         │                                      │
│              ("Hey Jarvis")         │         │                                      │
│                   │                 │         │                                      │
│          [wake word detected]       │         │                                      │
│                   │                 │         │                                      │
│          Play confirmation chime    │         │                                      │
│                   │                 │   WS    │                                      │
│          Connect WebSocket ─────────┼─────────┼──► Session starts (LISTENING)       │
│                   │                 │         │         │                            │
│          Stream audio ──────────────┼─────────┼──► Silero VAD ──► Whisper (small.en)│
│                   │                 │         │         │                            │
│                   │                 │         │         ▼                            │
│                   │                 │         │   Gemma 2 2B (4-bit)                 │
│                   │                 │         │         │                            │
│  ◄──────────────────────────────────┼─────────┼─── TRANSCRIPT + RESPONSE             │
│  Display response                   │         │                                      │
│  Return to wake word listening      │         │                                      │
└─────────────────────────────────────┘         └──────────────────────────────────────┘
```

## Features

- **Wake Word Detection**: Always-on listening for "Hey Jarvis" (custom "Hey Icarus" can be trained)
- **Voice Activity Detection (VAD)**: Silero VAD for accurate speech boundary detection
- **Speech-to-Text**: OpenAI Whisper (small.en) for fast, accurate transcription
- **Conversational AI**: Gemma 2 2B with custom "Icarus" persona - witty, concise, and helpful
- **Session Management**: Auto-timeout after 5s silence, or say "end session" to close
- **Low Latency**: Optimized for real-time voice interaction

## Requirements

### Backend (GPU Server)
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 4080 Super)
- ~5GB VRAM (Whisper + Gemma 4-bit quantized)

### Client (Laptop/Desktop)
- Python 3.10+
- Microphone
- Network connection to backend

## Setup

### Backend Setup

```bash
cd Backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Login to HuggingFace (required for Gemma)
huggingface-cli login

# Run the server
python main.py
```

### Client Setup

```bash
cd Client

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirement.txt

# Run the client
python client.py
```

## Usage

1. Start the backend server: `python main.py`
2. Start the client: `python client.py`
3. Say **"Hey Jarvis"** to activate
4. Speak your request
5. Wait for Icarus to respond
6. Say **"end session"** or wait 5 seconds of silence to close the session

## WebSocket Protocol

| Direction | Message | Description |
|-----------|---------|-------------|
| Client → Server | `WAKE_WORD:hey_icarus` | Wake word detected, start session |
| Client → Server | `<audio bytes>` | Raw PCM audio (16kHz, mono, int16) |
| Server → Client | `STATE:LISTENING` | Session started |
| Server → Client | `STATE:IDLE` | Session ended |
| Server → Client | `TRANSCRIPT:<text>` | User's transcribed speech |
| Server → Client | `RESPONSE:<text>` | Icarus's response |
| Server → Client | `ACK` | Audio chunk received |

## Project Structure

```
Project-Icarus/
├── Backend/
│   ├── main.py              # FastAPI server with VAD, Whisper, and LLM
│   ├── requirements.txt     # Backend dependencies
│   ├── train_wake_word.py   # Script to train custom wake word
│   └── models/              # Model storage directory
├── Client/
│   ├── client.py            # Audio capture and wake word detection
│   └── requirement.txt      # Client dependencies
└── frontend/                # Web UI (future)
```

## Configuration

Key settings in `Backend/main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SPEECH_THRESHOLD` | 0.5 | VAD sensitivity (lower = more sensitive) |
| `MIN_SILENCE_DURATION_MS` | 500 | Silence to end speech segment |
| `SESSION_TIMEOUT_SEC` | 5.0 | Timeout to auto-end session |
| `LLM_MAX_TOKENS` | 150 | Max response length |
| `LLM_TEMPERATURE` | 0.7 | Response creativity |

## Icarus Persona

Icarus is designed to be:
- Intelligent and efficient with dry humor
- Professional yet personable
- Concise for voice interaction (1-3 sentences typical)
- Occasionally sardonic, but never at the user's expense

## Future Roadmap

- [ ] Custom "Hey Icarus" wake word training
- [ ] Text-to-Speech (TTS) for voice responses
- [ ] Web UI for visual feedback
- [ ] Multi-user support
- [ ] Plugin system for extensibility

## License

MIT License