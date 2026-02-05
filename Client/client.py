# client.py
import asyncio
import pyaudio
import websockets
import logging
import numpy as np
import math
import sys
import json
import io
from enum import Enum, auto

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Conditional import for wake word (may not be installed yet)
try:
    from openwakeword.model import Model as WakeWordModel
    import openwakeword
    WAKE_WORD_AVAILABLE = True
except ImportError:
    WAKE_WORD_AVAILABLE = False
    print("⚠️ OpenWakeWord not installed. Run: pip install openwakeword onnxruntime")


# ═══════════════════════════════════════════════════════════════════════════════
# IPC Communication (for Electron parent process)
# ═══════════════════════════════════════════════════════════════════════════════
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ipc_emit(msg_type: str, payload: any = None):
    """
    Emit a structured message for the Electron parent process to consume.
    Format: IPC_JSON:{"type": "...", "payload": ...}
    """
    message = {"type": msg_type}
    if payload is not None:
        message["payload"] = payload
    print(f"IPC_JSON:{json.dumps(message, cls=NumpyEncoder)}")
    sys.stdout.flush()


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

# ═══════════════════════════════════════════════════════════════════════════════
# Wake Word Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Set to True to use custom "hey_icarus" model, False to use built-in "hey_jarvis"
USE_CUSTOM_WAKE_WORD = False  # TODO: Set to True once custom model is ready

# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Server URI - change this to your server's address (e.g., ws://192.168.1.100:8000/ws/audio)
import os
SERVER_URI = os.environ.get("ICARUS_SERVER", "ws://localhost:8000/ws/audio")
WS_PING_INTERVAL = 30  # Keep-alive ping interval (seconds)
WS_PING_TIMEOUT = 10   # Timeout for ping response
WS_RECONNECT_INTERVAL = 2.0  # Seconds between reconnection attempts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IcarusClient")


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════════════════════
class WebSocketManager:
    """
    Manages a persistent WebSocket connection to the server.
    Pre-connects at startup and maintains connection in background.
    Eliminates connection latency after wake word detection.
    """
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self._lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._reconnect_task = None
        self._running = False
    
    async def start(self):
        """Start the connection manager and establish initial connection."""
        self._running = True
        self._reconnect_task = asyncio.create_task(self._maintain_connection())
        # Wait for initial connection (with timeout)
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=10.0)
            logger.info("✓ WebSocket pre-connected and ready")
            ipc_emit("LOG", f"Connected to server: {self.uri}")
        except asyncio.TimeoutError:
            logger.warning("Initial connection timed out, will retry in background")
            ipc_emit("LOG", "Server connection pending, will connect on wake word")
    
    async def stop(self):
        """Stop the connection manager and close connection."""
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        if self.websocket:
            await self.websocket.close()
    
    async def _maintain_connection(self):
        """Background task to maintain persistent connection."""
        while self._running:
            if not self._is_connected():
                self._connected.clear()
                try:
                    await self._connect()
                except Exception as e:
                    logger.debug(f"Connection attempt failed: {e}")
                    await asyncio.sleep(WS_RECONNECT_INTERVAL)
                    continue
            await asyncio.sleep(0.5)

    def _is_connected(self) -> bool:
        """Check if websocket is connected (compatible with all websockets versions)."""
        if self.websocket is None:
            return False
        try:
            # Try .open property (works in most versions)
            if hasattr(self.websocket, 'open'):
                return self.websocket.open
            # Fallback: check state
            if hasattr(self.websocket, 'state'):
                from websockets.protocol import State
                return self.websocket.state == State.OPEN
            # Last resort: assume connected if websocket exists
            return True
        except Exception:
            return False

    async def _connect(self):
        """Establish WebSocket connection with optimized settings."""
        async with self._lock:
            logger.info(f"Connecting to {self.uri}...")
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                close_timeout=5,
                open_timeout=10,
                compression=None,  # Disable compression for lower latency
            )
            self._connected.set()
            logger.info(f"✓ Connected to {self.uri}")
    
    async def get_connection(self, timeout: float = 5.0):
        """
        Get the current WebSocket connection.
        Waits for connection if not yet established.
        Returns None if connection unavailable within timeout.
        """
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            if self._is_connected():
                return self.websocket
        except asyncio.TimeoutError:
            pass
        return None
    
    async def reconnect(self):
        """
        Force a reconnection (call after session ends since server may close it).
        """
        async with self._lock:
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass
                self.websocket = None
            self._connected.clear()
        # Background task will reconnect automatically


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
        import os
        
        if USE_CUSTOM_WAKE_WORD:
            # Use custom trained "hey_icarus" model
            model_path = os.path.join(os.path.dirname(__file__), "models", "hey_icarus.onnx")
            if os.path.exists(model_path):
                model = WakeWordModel(wakeword_models=[model_path])
                logger.info("✓ Custom 'Hey Icarus' wake word model loaded!")
                ipc_emit("LOG", "Custom 'Hey Icarus' wake word model loaded")
                return model
            else:
                logger.warning("Custom model file not found, falling back to hey_jarvis")
        
        # Use built-in "hey_jarvis" model
        openwakeword.utils.download_models()
        model = WakeWordModel(wakeword_models=["hey_jarvis"])
        logger.info("Using built-in 'hey_jarvis' wake word model")
        ipc_emit("LOG", "Using 'hey_jarvis' wake word model")
        
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
    ipc_emit("STATE", "IDLE")
    
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
                ipc_emit("WAKE_WORD", {"model": model_name, "score": round(score, 2)})
                return True
        
        # Small yield to prevent blocking
        await asyncio.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Audio Streaming Session
# ═══════════════════════════════════════════════════════════════════════════════
async def run_session(stream, p, start_chime, end_chime, ws_manager: WebSocketManager):
    """Run a single voice session after wake word detection."""
    
    # Play confirmation sound IMMEDIATELY (don't wait for connection)
    play_sound(start_chime, p)
    
    ipc_emit("STATE", "CONNECTING")
    
    try:
        # Try to get pre-established connection (should be instant if connected)
        websocket = await ws_manager.get_connection(timeout=3.0)
        
        if websocket is None:
            # Fallback: connect on-demand if pre-connection failed
            logger.info(f"Pre-connection unavailable, connecting now...")
            websocket = await websockets.connect(
                SERVER_URI,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                compression=None,
            )
            logger.info("Connected (on-demand).")
        else:
            logger.info("Using pre-established connection (instant!)")
        
        # Notify server that wake word was detected
        await websocket.send("WAKE_WORD:hey_icarus")
        
        # Wait for server to acknowledge
        response = await websocket.recv()
        if response != "STATE:LISTENING":
            logger.warning(f"Unexpected response: {response}")
        
        logger.info("🎙️ Streaming audio... (say 'end session' or wait 5s silence to stop)")
        ipc_emit("STATE", "LISTENING")
        
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
                ipc_emit("TRANSCRIPT", transcript)
            
            elif response.startswith("RESPONSE:"):
                llm_response = response.split(":", 1)[1]
                print(f"🤖 Icarus: {llm_response}")
                ipc_emit("RESPONSE", llm_response)
                ipc_emit("STATE", "SPEAKING")
            
            elif response == "AUDIO_END":
                # Play accumulated TTS audio
                if audio_buffer:
                    logger.info("🔊 Playing TTS response...")
                    await asyncio.get_event_loop().run_in_executor(
                        None, play_tts_audio, audio_buffer, p
                    )
                    audio_buffer = b""
                    ipc_emit("STATE", "LISTENING")
                
            elif response == "STATE:IDLE":
                logger.info("Session ended by server")
                ipc_emit("STATE", "IDLE")
                break
                
            elif response != "ACK":
                logger.debug(f"Server: {response}")
        
        # Play end sound
        play_sound(end_chime, p)
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
        ipc_emit("STATE", "IDLE")
        play_sound(end_chime, p)
    except Exception as e:
        logger.error(f"Session error: {e}")
        ipc_emit("ERROR", str(e))
        play_sound(end_chime, p)
    finally:
        # Trigger reconnection for next session (server closes after each session)
        await ws_manager.reconnect()


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
    
    # Pre-establish WebSocket connection in background
    logger.info(f"Connecting to server: {SERVER_URI}")
    ws_manager = WebSocketManager(SERVER_URI)
    await ws_manager.start()
    
    logger.info("Icarus Client started.")
    ipc_emit("READY", True)
    
    try:
        while True:
            # Wait for wake word
            await listen_for_wake_word(wake_model, stream, p)
            
            # Run voice session (uses pre-established connection)
            await run_session(stream, p, start_chime, end_chime, ws_manager)
            
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
            ipc_emit("STATE", "IDLE")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await ws_manager.stop()
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")