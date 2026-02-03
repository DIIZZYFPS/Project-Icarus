# client.py
import asyncio
import pyaudio
import websockets
import logging
import numpy as np
import math
from enum import Enum, auto

# Conditional import for wake word (may not be installed yet)
try:
    from openwakeword.model import Model as WakeWordModel
    import openwakeword
    WAKE_WORD_AVAILABLE = True
except ImportError:
    WAKE_WORD_AVAILABLE = False
    print("⚠️ OpenWakeWord not installed. Run: pip install openwakeword onnxruntime")


# ═══════════════════════════════════════════════════════════════════════════════
# Client State
# ═══════════════════════════════════════════════════════════════════════════════
class ClientState(Enum):
    IDLE = auto()           # Listening for wake word
    CONNECTING = auto()     # Connecting to server
    STREAMING = auto()      # Streaming audio to server


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Configuration (Standard for Whisper/VAD)
# ═══════════════════════════════════════════════════════════════════════════════
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TTS_RATE = 24000  # Edge TTS outputs at 24kHz
CHUNK = 4096  # 256ms of audio per packet
WAKE_WORD_CHUNK = 1280  # 80ms chunks for OpenWakeWord
WAKE_WORD_THRESHOLD = 0.5  # Detection threshold
POST_SESSION_COOLDOWN = 1.5  # Seconds to wait after session ends before listening again

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IcarusClient")


# ═══════════════════════════════════════════════════════════════════════════════
# Sound Generation (simple confirmation tones)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tone(frequency: float, duration: float, sample_rate: int = 16000) -> bytes:
    """Generate a simple sine wave tone."""
    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        # Apply fade in/out to avoid clicks
        t = i / sample_rate
        fade = min(1.0, min(t / 0.01, (duration - t) / 0.01))
        sample = int(32767 * 0.3 * fade * math.sin(2 * math.pi * frequency * t))
        samples.append(sample)
    return np.array(samples, dtype=np.int16).tobytes()


def generate_start_chime() -> bytes:
    """Two-tone ascending chime for wake word confirmation."""
    tone1 = generate_tone(800, 0.1)   # Lower tone
    tone2 = generate_tone(1200, 0.15)  # Higher tone
    return tone1 + tone2


def generate_end_chime() -> bytes:
    """Two-tone descending chime for session end."""
    tone1 = generate_tone(1200, 0.1)  # Higher tone
    tone2 = generate_tone(800, 0.15)   # Lower tone
    return tone1 + tone2


def play_sound(audio_bytes: bytes, p: pyaudio.PyAudio):
    """Play audio bytes through speakers."""
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    stream.write(audio_bytes)
    stream.stop_stream()
    stream.close()


def play_tts_audio(audio_bytes: bytes, p: pyaudio.PyAudio):
    """Play TTS audio at 22050 Hz sample rate."""
    if not audio_bytes:
        return
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=TTS_RATE,
                    output=True)
    stream.write(audio_bytes)
    stream.stop_stream()
    stream.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Wake Word Detection
# ═══════════════════════════════════════════════════════════════════════════════
def init_wake_word_model():
    """Initialize OpenWakeWord model."""
    if not WAKE_WORD_AVAILABLE:
        return None
    
    try:
        # Use custom trained "hey_icarus" model
        import os
        model_path = os.path.join(os.path.dirname(__file__), "models", "hey_icarus.onnx")
        
        if os.path.exists(model_path):
            model = WakeWordModel(wakeword_models=[model_path])
            logger.info("✓ Custom 'Hey Icarus' wake word model loaded!")
        else:
            # Fallback to built-in model if custom not found
            openwakeword.utils.download_models()
            model = WakeWordModel(wakeword_models=["hey_jarvis"])
            logger.warning("Custom model not found, using 'hey_jarvis' fallback")
        
        return model
    except Exception as e:
        logger.warning(f"Failed to load wake word model: {e}")
        return None


async def listen_for_wake_word(wake_model, stream, p) -> bool:
    """
    Listen for wake word. Returns True when detected.
    Runs in a loop reading small chunks for responsive detection.
    """
    if wake_model is None:
        # No wake word model - auto-activate for testing
        logger.info("No wake word model - press Enter to activate...")
        await asyncio.get_event_loop().run_in_executor(None, input)
        return True
    
    logger.info("🎧 Listening for wake word ('Hey Jarvis')...")
    
    while True:
        # Read smaller chunks for wake word detection
        data = stream.read(WAKE_WORD_CHUNK, exception_on_overflow=False)
        
        # Convert to int16 numpy array
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # Run wake word prediction
        prediction = wake_model.predict(audio_array)
        
        # Check if any wake word detected
        for model_name, score in prediction.items():
            if score > WAKE_WORD_THRESHOLD:
                logger.info(f"🎯 Wake word detected: {model_name} (score: {score:.2f})")
                return True
        
        # Small yield to prevent blocking
        await asyncio.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Audio Streaming Session
# ═══════════════════════════════════════════════════════════════════════════════
async def run_session(stream, p, start_chime, end_chime):
    """Run a single voice session after wake word detection."""
    
    # Play confirmation sound
    play_sound(start_chime, p)
    
    uri = "ws://localhost:8000/ws/audio"
    logger.info(f"Connecting to Icarus Brain at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected. Sending wake word notification...")
            
            # Notify server that wake word was detected
            await websocket.send("WAKE_WORD:hey_icarus")
            
            # Wait for server to acknowledge
            response = await websocket.recv()
            if response != "STATE:LISTENING":
                logger.warning(f"Unexpected response: {response}")
            
            logger.info("🎙️ Streaming audio... (say 'end session' or wait 5s silence to stop)")
            
            audio_buffer = b""  # Buffer for incoming TTS audio
            
            while True:
                # Read audio from microphone
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Send to server
                await websocket.send(data)
                
                # Receive response (may be text or binary audio)
                response = await websocket.recv()
                
                # Handle binary audio data
                if isinstance(response, bytes):
                    audio_buffer += response
                    continue
                
                # Handle different text response types
                if response.startswith("TRANSCRIPT:"):
                    transcript = response.split(":", 1)[1]
                    print(f"📝 You said: {transcript}")
                
                elif response.startswith("RESPONSE:"):
                    llm_response = response.split(":", 1)[1]
                    print(f"🤖 Icarus: {llm_response}")
                
                elif response == "AUDIO_END":
                    # Play accumulated TTS audio
                    if audio_buffer:
                        logger.info("🔊 Playing TTS response...")
                        await asyncio.get_event_loop().run_in_executor(
                            None, play_tts_audio, audio_buffer, p
                        )
                        audio_buffer = b""
                    
                elif response == "STATE:IDLE":
                    logger.info("Session ended by server")
                    break
                    
                elif response != "ACK":
                    logger.debug(f"Server: {response}")
            
            # Play end sound
            play_sound(end_chime, p)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
        play_sound(end_chime, p)
    except Exception as e:
        logger.error(f"Session error: {e}")
        play_sound(end_chime, p)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def main():
    """Main loop: wake word → session → repeat."""
    
    p = pyaudio.PyAudio()
    
    # Open microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=WAKE_WORD_CHUNK)
    
    # Generate confirmation sounds
    start_chime = generate_start_chime()
    end_chime = generate_end_chime()
    
    # Initialize wake word model
    wake_model = init_wake_word_model()
    
    logger.info("Icarus Client started.")
    
    try:
        while True:
            # Wait for wake word
            await listen_for_wake_word(wake_model, stream, p)
            
            # Run voice session
            await run_session(stream, p, start_chime, end_chime)
            
            # Cooldown period to prevent chime from triggering wake word
            logger.info(f"Cooldown for {POST_SESSION_COOLDOWN}s...")
            await asyncio.sleep(POST_SESSION_COOLDOWN)
            
            # Flush any audio that accumulated during cooldown
            try:
                while stream.get_read_available() > 0:
                    stream.read(stream.get_read_available(), exception_on_overflow=False)
            except Exception:
                pass
            
            # Reset wake word model state if it has one
            if wake_model is not None and hasattr(wake_model, 'reset'):
                wake_model.reset()
            
            logger.info("Ready for next wake word...\n")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")