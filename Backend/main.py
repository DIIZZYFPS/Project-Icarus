from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import torch
import numpy as np
import asyncio
import time
from enum import Enum, auto
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel

# Set up logging to track the "Split Brain" connection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IcarusBrain")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev, tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════════════
class SessionState(Enum):
    IDLE = auto()           # Waiting for wake word from client
    LISTENING = auto()      # Active session, processing audio
    PROCESSING = auto()     # Transcribing/responding


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000  # Must match client's sample rate
VAD_WINDOW_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz
SPEECH_THRESHOLD = 0.5  # Probability threshold for speech detection
MIN_SILENCE_DURATION_MS = 500  # Silence duration to consider speech ended
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to keep
SESSION_TIMEOUT_SEC = 5.0  # End session after 5s of no speech
END_SESSION_PHRASES = ["end session", "and session"]  # Fuzzy match for mishearing


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("Loading Silero VAD model...")
vad_model = load_silero_vad()
logger.info("Silero VAD model loaded successfully.")

logger.info("Loading Whisper model (small.en)...")
whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
logger.info("Whisper model loaded successfully.")


@app.get("/")
async def read_root():
    return {"Status": "Icarus Brain Online"}


def bytes_to_float32_tensor(audio_bytes: bytes) -> torch.Tensor:
    """Convert raw PCM bytes (int16) to float32 tensor normalized to [-1, 1]."""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return torch.from_numpy(audio_float32)


def process_vad_on_chunk(audio_tensor: torch.Tensor) -> float:
    """
    Process audio through VAD by splitting into 512-sample windows.
    Returns the maximum speech probability across all windows.
    """
    num_samples = len(audio_tensor)
    max_prob = 0.0
    
    # Process in 512-sample windows as required by Silero VAD
    for i in range(0, num_samples - VAD_WINDOW_SIZE + 1, VAD_WINDOW_SIZE):
        window = audio_tensor[i:i + VAD_WINDOW_SIZE]
        prob = vad_model(window, SAMPLE_RATE).item()
        max_prob = max(max_prob, prob)
    
    return max_prob


def transcribe_audio_sync(audio_bytes: bytes) -> str:
    """
    Transcribe PCM audio bytes using Whisper (synchronous).
    Called via asyncio.to_thread() to avoid blocking.
    """
    # Convert int16 PCM to float32 numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Transcribe (VAD already filtered, so disable internal VAD)
    segments, info = whisper_model.transcribe(
        audio_float32,
        beam_size=5,
        language="en",
        vad_filter=False,
    )
    
    # Combine all segments
    transcript = " ".join(segment.text.strip() for segment in segments)
    return transcript


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Async wrapper for Whisper transcription."""
    return await asyncio.to_thread(transcribe_audio_sync, audio_bytes)


def check_end_session(transcript: str) -> bool:
    """Check if transcript contains end session phrase."""
    transcript_lower = transcript.lower().strip()
    for phrase in END_SESSION_PHRASES:
        if phrase in transcript_lower:
            return True
    return False


# This is the endpoint your client.py is trying to hit
@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via SSH Tunnel.")
    
    # Per-connection state
    session_state = SessionState.IDLE
    audio_buffer = []  # Accumulates audio chunks during speech
    is_speaking = False
    silence_samples = 0
    samples_for_silence = int(SAMPLE_RATE * MIN_SILENCE_DURATION_MS / 1000)
    last_speech_time = time.time()
    
    try:
        while True:
            # Use timeout to check for session expiry
            try:
                # Wait for data with a short timeout to allow checking session timeout
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                # Check session timeout when in LISTENING state
                if session_state == SessionState.LISTENING:
                    if time.time() - last_speech_time > SESSION_TIMEOUT_SEC:
                        logger.info("⏱️ Session timeout - no speech for 5 seconds")
                        session_state = SessionState.IDLE
                        await websocket.send_text("STATE:IDLE")
                        audio_buffer = []
                        is_speaking = False
                continue
            
            # Handle different message types
            if "text" in data:
                message = data["text"]
                if message.startswith("WAKE_WORD:"):
                    # Client detected wake word, start listening
                    wake_word = message.split(":", 1)[1]
                    logger.info(f"🎯 Wake word detected: {wake_word}")
                    session_state = SessionState.LISTENING
                    last_speech_time = time.time()
                    await websocket.send_text("STATE:LISTENING")
                continue
            
            if "bytes" not in data:
                continue
                
            audio_data = data["bytes"]
            
            # Only process audio when in LISTENING state
            if session_state != SessionState.LISTENING:
                await websocket.send_text("ACK")
                continue
            
            # Convert bytes to tensor for VAD processing
            audio_tensor = bytes_to_float32_tensor(audio_data)
            
            # Run VAD on this chunk
            speech_prob = process_vad_on_chunk(audio_tensor)
            
            # State machine for speech detection
            if speech_prob >= SPEECH_THRESHOLD:
                # Speech detected
                if not is_speaking:
                    logger.info(f"🎤 Speech started (prob: {speech_prob:.2f})")
                    is_speaking = True
                
                audio_buffer.append(audio_data)
                silence_samples = 0
                last_speech_time = time.time()
                
            else:
                # Silence detected
                if is_speaking:
                    audio_buffer.append(audio_data)
                    silence_samples += len(audio_tensor)
                    
                    # Check if silence duration exceeded threshold
                    if silence_samples >= samples_for_silence:
                        total_bytes = sum(len(chunk) for chunk in audio_buffer)
                        duration_ms = (total_bytes / 2) / SAMPLE_RATE * 1000
                        
                        if duration_ms >= MIN_SPEECH_DURATION_MS:
                            complete_audio = b"".join(audio_buffer)
                            logger.info(f"🔊 Speech ended: {duration_ms:.0f}ms, {len(complete_audio)} bytes")
                            
                            # Transcribe with Whisper
                            session_state = SessionState.PROCESSING
                            transcript = await transcribe_audio(complete_audio)
                            logger.info(f"📝 Transcript: {transcript}")
                            
                            # Check for end session command
                            if check_end_session(transcript):
                                logger.info("👋 End session command detected")
                                await websocket.send_text(f"TRANSCRIPT:{transcript}")
                                await websocket.send_text("STATE:IDLE")
                                session_state = SessionState.IDLE
                            else:
                                await websocket.send_text(f"TRANSCRIPT:{transcript}")
                                session_state = SessionState.LISTENING
                                last_speech_time = time.time()
                        else:
                            logger.info(f"⏭️ Discarded short segment: {duration_ms:.0f}ms")
                        
                        # Reset speech detection state
                        audio_buffer = []
                        is_speaking = False
                        silence_samples = 0
            
            await websocket.send_text("ACK")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is crucial so it listens on the Tailscale/SSH interface, not just local loopback
    uvicorn.run(app, host="0.0.0.0", port=8000)