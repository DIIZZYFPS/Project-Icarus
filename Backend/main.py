from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

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
# VAD Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000  # Must match client's sample rate
VAD_WINDOW_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz
SPEECH_THRESHOLD = 0.5  # Probability threshold for speech detection
MIN_SILENCE_DURATION_MS = 500  # Silence duration to consider speech ended
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to keep

# Load Silero VAD model at startup
logger.info("Loading Silero VAD model...")
vad_model = load_silero_vad()
logger.info("Silero VAD model loaded successfully.")


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


# This is the endpoint your client.py is trying to hit
@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via SSH Tunnel.")
    
    # Per-connection state for VAD
    audio_buffer = []  # Accumulates audio chunks during speech
    is_speaking = False
    silence_samples = 0
    samples_for_silence = int(SAMPLE_RATE * MIN_SILENCE_DURATION_MS / 1000)
    
    try:
        while True:
            # 1. Receive raw audio bytes from the "Ears" (Laptop)
            data = await websocket.receive_bytes()
            
            # 2. Convert bytes to tensor for VAD processing
            audio_tensor = bytes_to_float32_tensor(data)
            
            # 3. Run VAD on this chunk (splits into 512-sample windows internally)
            speech_prob = process_vad_on_chunk(audio_tensor)
            
            # 4. State machine for speech detection
            if speech_prob >= SPEECH_THRESHOLD:
                # Speech detected
                if not is_speaking:
                    logger.info(f"🎤 Speech started (prob: {speech_prob:.2f})")
                    is_speaking = True
                
                audio_buffer.append(data)
                silence_samples = 0
                
            else:
                # Silence detected
                if is_speaking:
                    audio_buffer.append(data)  # Keep buffering during silence gap
                    silence_samples += len(audio_tensor)
                    
                    # Check if silence duration exceeded threshold
                    if silence_samples >= samples_for_silence:
                        # Speech segment complete!
                        total_bytes = sum(len(chunk) for chunk in audio_buffer)
                        duration_ms = (total_bytes / 2) / SAMPLE_RATE * 1000
                        
                        if duration_ms >= MIN_SPEECH_DURATION_MS:
                            # Combine all chunks into one audio segment
                            complete_audio = b"".join(audio_buffer)
                            logger.info(f"🔊 Speech ended: {duration_ms:.0f}ms, {len(complete_audio)} bytes")
                            
                            # TODO: Send complete_audio to Whisper for transcription
                            # For now, notify client that speech segment was captured
                            await websocket.send_text(f"SPEECH_SEGMENT:{len(complete_audio)}")
                        else:
                            logger.info(f"⏭️ Discarded short segment: {duration_ms:.0f}ms")
                        
                        # Reset state
                        audio_buffer = []
                        is_speaking = False
                        silence_samples = 0
            
            # 5. Send ACK (keeps the socket alive)
            await websocket.send_text("ACK")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is crucial so it listens on the Tailscale/SSH interface, not just local loopback
    uvicorn.run(app, host="0.0.0.0", port=8000)