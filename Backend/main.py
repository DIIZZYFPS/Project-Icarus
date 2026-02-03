from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import torch
import numpy as np
import asyncio
import time
import os
import wave
import io
from pathlib import Path
from enum import Enum, auto
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from piper import PiperVoice

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
    PROCESSING = auto()     # Transcribing
    GENERATING = auto()     # LLM generating response


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000  # Must match client's sample rate
VAD_WINDOW_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz
SPEECH_THRESHOLD = 0.5  # Probability threshold for speech detection
MIN_SILENCE_DURATION_MS = 500  # Silence duration to consider speech ended
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to keep
SESSION_TIMEOUT_SEC = 5.0  # End session after 5s of no speech
END_SESSION_PHRASES = ["end session", "and session", "that's all", "that is all"]  # Fuzzy match for mishearing

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════════
LLM_MODEL_ID = "google/gemma-2-2b-it"  # Hugging Face model ID
LLM_MAX_TOKENS = 150  # Keep responses concise for voice
LLM_TEMPERATURE = 0.7
MAX_CONVERSATION_TURNS = 10  # Keep last N exchanges

# ═══════════════════════════════════════════════════════════════════════════════
# TTS Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TTS_MODEL_PATH = Path(__file__).parent / "models" / "tts" / "en_GB-alan-medium.onnx"
TTS_SAMPLE_RATE = 22050  # Piper outputs at 22050 Hz

ICARUS_SYSTEM_PROMPT = """You are Icarus, an advanced AI assistant created to help your user navigate their digital world with precision and wit. Named after the mythological figure who dared to fly—though you've learned to respect your limits while still reaching for the sky.

Personality traits:
- Intelligent and efficient, with a dry sense of humor
- Professional yet personable—think trusted colleague, not cold machine
- Occasionally sardonic, but never at your user's expense
- Proactive in offering solutions, not just answering questions
- Concise in speech—you understand brevity is valued in voice interaction

Communication style:
- Keep responses short and natural for spoken delivery (1-3 sentences typical)
- Use conversational language, not formal or robotic phrasing
- Light wit is welcome; lengthy monologues are not
- When asked complex questions, give the essential answer first, offer to elaborate if needed

You assist with tasks, answer questions, provide information, and occasionally remind your user that while ambition is admirable, even you know when to pull back from the sun."""


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("Loading Silero VAD model...")
vad_model = load_silero_vad()
logger.info("Silero VAD model loaded successfully.")

logger.info("Loading Whisper model (small.en)...")
whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
logger.info("Whisper model loaded successfully.")

# Warmup Whisper with dummy transcription to compile CUDA kernels
logger.info("Warming up Whisper model (first inference is slow)...")
_warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second silence
_warmup_segments, _ = whisper_model.transcribe(_warmup_audio, language="en")
list(_warmup_segments)  # Force generator execution
logger.info("Whisper warmup complete.")

# Load LLM (Gemma via Transformers)
try:
    logger.info(f"Loading LLM from {LLM_MODEL_ID}...")
    
    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    
    # Load model with quantization
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    logger.info("LLM loaded successfully.")
    
    # Warmup LLM
    logger.info("Warming up LLM...")
    _warmup_inputs = llm_tokenizer("Hello", return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        llm_model.generate(**_warmup_inputs, max_new_tokens=5)
    logger.info("LLM warmup complete.")
    
except Exception as e:
    logger.warning(f"Failed to load LLM: {e}")
    logger.warning("LLM responses will be disabled.")
    llm_model = None
    llm_tokenizer = None

# Load TTS (Piper)
try:
    logger.info(f"Loading TTS model from {TTS_MODEL_PATH}...")
    tts_voice = PiperVoice.load(str(TTS_MODEL_PATH))
    logger.info("TTS model loaded successfully.")
    
    # Warmup TTS
    logger.info("Warming up TTS...")
    _warmup_audio = list(tts_voice.synthesize("Hello."))
    logger.info("TTS warmup complete.")
except Exception as e:
    logger.warning(f"Failed to load TTS: {e}")
    logger.warning("TTS responses will be disabled.")
    tts_voice = None


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
    logger.info("🔄 Starting Whisper transcription...")
    start_time = time.time()
    
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
    
    # Force generator to execute (this is where actual transcription happens)
    segments_list = list(segments)
    
    elapsed = time.time() - start_time
    logger.info(f"🔄 Whisper completed in {elapsed:.2f}s, {len(segments_list)} segments")
    
    # Combine all segments
    transcript = " ".join(segment.text.strip() for segment in segments_list)
    return transcript


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Async wrapper for Whisper transcription."""
    return await asyncio.to_thread(transcribe_audio_sync, audio_bytes)


def synthesize_speech_sync(text: str) -> bytes:
    """
    Synthesize text to speech using Piper (synchronous).
    Returns raw PCM audio bytes (int16, 22050 Hz).
    """
    if tts_voice is None:
        return b""
    
    logger.info(f"🔊 Synthesizing speech: {text[:50]}...")
    start_time = time.time()
    
    try:
        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(TTS_SAMPLE_RATE)
            tts_voice.synthesize(text, wav_file)
        
        # Extract raw PCM from WAV (skip 44-byte header)
        wav_buffer.seek(44)
        audio_data = wav_buffer.read()
        
        elapsed = time.time() - start_time
        duration_sec = len(audio_data) / 2 / TTS_SAMPLE_RATE
        logger.info(f"🔊 TTS completed in {elapsed:.2f}s ({duration_sec:.1f}s audio)")
        
        return audio_data
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""


async def synthesize_speech(text: str) -> bytes:
    """Async wrapper for TTS synthesis."""
    return await asyncio.to_thread(synthesize_speech_sync, text)


def check_end_session(transcript: str) -> bool:
    """Check if transcript contains end session phrase."""
    transcript_lower = transcript.lower().strip()
    for phrase in END_SESSION_PHRASES:
        if phrase in transcript_lower:
            return True
    return False


def generate_response_sync(user_message: str, conversation_history: list) -> str:
    """
    Generate LLM response (synchronous).
    Called via asyncio.to_thread() to avoid blocking.
    """
    if llm_model is None or llm_tokenizer is None:
        return "I apologize, but my language model isn't loaded. I can hear you, but I can't formulate a proper response."
    
    logger.info("🧠 Generating LLM response...")
    start_time = time.time()
    
    try:
        # Build conversation for Gemma chat format
        messages = [{"role": "user", "content": ICARUS_SYSTEM_PROMPT + "\n\nAcknowledge this persona briefly."}]
        messages.append({"role": "assistant", "content": "Understood. I'm Icarus, ready to assist with precision and perhaps a touch of wit. What do you need?"})
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Apply chat template
        prompt = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = llm_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        elapsed = time.time() - start_time
        logger.info(f"🧠 LLM completed in {elapsed:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "I seem to have hit some turbulence. Could you repeat that?"


async def generate_response(user_message: str, conversation_history: list) -> str:
    """Async wrapper for LLM response generation."""
    return await asyncio.to_thread(generate_response_sync, user_message, conversation_history)


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
    conversation_history = []  # Tracks conversation for LLM context
    
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
                            
                            # Send transcript to client immediately
                            await websocket.send_text(f"TRANSCRIPT:{transcript}")
                            
                            # Check for end session command
                            if check_end_session(transcript):
                                logger.info("👋 End session command detected")
                                await websocket.send_text("STATE:IDLE")
                                session_state = SessionState.IDLE
                                conversation_history = []  # Clear history on session end
                            else:
                                # Generate LLM response
                                session_state = SessionState.GENERATING
                                llm_response = await generate_response(transcript, conversation_history)
                                logger.info(f"🤖 Response: {llm_response}")
                                
                                # Update conversation history
                                conversation_history.append({"role": "user", "content": transcript})
                                conversation_history.append({"role": "assistant", "content": llm_response})
                                
                                # Trim history if too long
                                if len(conversation_history) > MAX_CONVERSATION_TURNS * 2:
                                    conversation_history = conversation_history[-MAX_CONVERSATION_TURNS * 2:]
                                
                                # Send response to client
                                await websocket.send_text(f"RESPONSE:{llm_response}")
                                
                                # Synthesize and send audio
                                if tts_voice is not None:
                                    audio_data = await synthesize_speech(llm_response)
                                    if audio_data:
                                        await websocket.send_bytes(audio_data)
                                        await websocket.send_text("AUDIO_END")
                                
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