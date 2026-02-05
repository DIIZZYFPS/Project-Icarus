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
# from faster_whisper import WhisperModel
from funasr import AutoModel
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from kokoro import KPipeline
import re

# Set up logging to track the "Split Brain" connection
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("IcarusBrain")
logger.setLevel(logging.INFO)

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
SESSION_TIMEOUT_SEC = 100 # End session after 100s of no speech
END_SESSION_PHRASES = ["end session", "and session", "that's all", "that is all"]  # Fuzzy match for mishearing

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════════
LLM_MODEL_ID = "pytorch/gemma-3-12b-it-INT4"  # TorchAO INT4 pre-quantized (1.7x faster)
LLM_MAX_TOKENS = 150  # Keep responses concise for voice
LLM_TEMPERATURE = 0.7
MAX_CONVERSATION_TURNS = 10  # Keep last N exchanges

# ═══════════════════════════════════════════════════════════════════════════════
# TTS Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TTS_VOICE = "bm_george"  # British male voice
TTS_SAMPLE_RATE = 24000  # Kokoro outputs at 24kHz

# ═══════════════════════════════════════════════════════════════════════════════
# Emotion Mapping (SenseVoice emotions to natural descriptions for LLM)
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_MAP = {
    "HAPPY": "happy and upbeat",
    "SAD": "sad or melancholic",
    "ANGRY": "frustrated or angry",
    "SURPRISED": "surprised or caught off guard",
    "FEARFUL": "anxious or worried",
    "DISGUSTED": "displeased",
    "NEUTRAL": None,  # No special context needed
    "UNKNOWN": None,  # Can't determine emotion
}

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

logger.info("Loading SenseVoice model ...")
sense_voice_model = AutoModel(
    model = "iic/SenseVoiceSmall",
    device = "cuda",
    disable_update = True,
    disable_pbar = True,
)
logger.info("SenseVoice model loaded successfully.")

# Warmup SenseVoice with dummy transcription to compile CUDA kernels
logger.info("Warming up SenseVoice model (first inference is slow)...")

try:
    _warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second silence
    sense_voice_model.generate(
        input = _warmup_audio,
        cache = {},
        language = "auto",
        use_itn = True,
    )
    logger.info("SenseVoice warmup complete.")
except Exception as e:
    logger.warning(f"Failed to warmup SenseVoice model: {e}")

# Load LLM (Gemma 3 via Transformers - TorchAO INT4 pre-quantized)
try:
    logger.info(f"Loading LLM from {LLM_MODEL_ID}...")
    
    # Use AutoProcessor for Gemma 3 (handles multimodal architecture)
    # Load from base model for processor compatibility
    llm_processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")
    
    # Load TorchAO INT4 pre-quantized model (1.7x faster than bitsandbytes)
    llm_model = Gemma3ForConditionalGeneration.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    
    logger.info("LLM loaded successfully.")
    
    # Warmup LLM with Gemma 3 message format
    logger.info("Warming up LLM...")
    _warmup_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    _warmup_inputs = llm_processor.apply_chat_template(
        _warmup_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(llm_model.device)
    with torch.no_grad():
        llm_model.generate(**_warmup_inputs, max_new_tokens=5)
    logger.info("LLM warmup complete.")
    
except Exception as e:
    logger.warning(f"Failed to load LLM: {e}")
    logger.warning("LLM responses will be disabled.")
    llm_model = None
    llm_processor = None

# TTS (Kokoro - local TTS)
try:
    logger.info(f"Loading Kokoro TTS with voice: {TTS_VOICE}...")
    kokoro_pipeline = KPipeline(lang_code='b')  # 'b' = British English
    logger.info("Kokoro TTS loaded successfully.")
    
    # Warmup TTS
    logger.info("Warming up Kokoro TTS...")
    for _, _, _audio in kokoro_pipeline("Hello.", voice=TTS_VOICE):
        pass  # Just run through to warm up
    logger.info("Kokoro TTS warmup complete.")
    tts_enabled = True
except Exception as e:
    logger.warning(f"Failed to load Kokoro TTS: {e}")
    logger.warning("TTS responses will be disabled.")
    kokoro_pipeline = None
    tts_enabled = False


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


def parse_sensevoice_output(raw_text: str) -> tuple[str, str | None]:
    """
    Parse SenseVoice output to extract clean text and emotion.
    
    SenseVoice format: <|en|><|EMOTION|><|Event|><|withitn|>Actual text here.
    
    Returns: (clean_text, emotion) where emotion is the detected emotion or None
    """
    # Pattern to match all tags like <|something|>
    tag_pattern = r'<\|([^|]+)\|>'
    
    # Find all tags
    tags = re.findall(tag_pattern, raw_text)
    
    # Remove all tags to get clean text
    clean_text = re.sub(tag_pattern, '', raw_text).strip()
    
    # Look for emotion in tags (check against our emotion map)
    emotion = None
    for tag in tags:
        tag_upper = tag.upper()
        if tag_upper in EMOTION_MAP:
            emotion = tag_upper
            break
    
    return clean_text, emotion


def transcribe_audio_sync(audio_bytes: bytes) -> tuple[str, str | None]:
    """
    Transcribe PCM audio bytes using SenseVoice (synchronous).
    Called via asyncio.to_thread() to avoid blocking.
    
    Returns: (clean_text, emotion) tuple
    """
    logger.info("🔄 Starting SenseVoice transcription...")
    start_time = time.time()
    
    # Convert int16 PCM to float32 numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Transcribe (VAD already filtered, so disable internal VAD)
    res = sense_voice_model.generate(
        input = audio_float32,
        cache = {},
        language = "en",
        use_itn = True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15
    )
    
    elapsed = time.time() - start_time

    raw_text = ""
    if isinstance(res, list) and len(res) > 0:
        raw_text = res[0].get("text", "")

    logger.info(f"🔄 SenseVoice raw output in {elapsed:.2f}s: '{raw_text}'")
    
    # Parse to extract clean text and emotion
    clean_text, emotion = parse_sensevoice_output(raw_text)
    
    if emotion:
        logger.info(f"🎭 Detected emotion: {emotion}")
    
    return clean_text, emotion


async def transcribe_audio(audio_bytes: bytes) -> tuple[str, str | None]:
    """Async wrapper for SenseVoice transcription. Returns (clean_text, emotion)."""
    return await asyncio.to_thread(transcribe_audio_sync, audio_bytes)


def synthesize_speech_sync(text: str) -> bytes:
    """
    Synthesize text to speech using Kokoro (synchronous).
    Returns raw PCM audio bytes (int16, 24kHz).
    """
    if not tts_enabled or kokoro_pipeline is None:
        return b""
    
    logger.info(f"🔊 Synthesizing speech: {text[:50]}...")
    start_time = time.time()
    
    try:
        # Collect all audio chunks from Kokoro
        audio_chunks = []
        for _, _, audio in kokoro_pipeline(text, voice=TTS_VOICE, speed=1.0):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            logger.warning("Kokoro returned no audio")
            return b""
        
        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)
        
        # Convert float32 [-1, 1] to int16 PCM
        audio_int16 = (full_audio * 32767).astype(np.int16)
        pcm_data = audio_int16.tobytes()
        
        elapsed = time.time() - start_time
        duration_sec = len(pcm_data) / 2 / TTS_SAMPLE_RATE
        logger.info(f"🔊 TTS completed in {elapsed:.2f}s ({duration_sec:.1f}s audio)")
        
        return pcm_data
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""


async def synthesize_speech(text: str, websocket) -> bool:
    """
    Synthesize text to speech and stream to client.
    Returns True if audio was sent successfully.
    """
    if not tts_enabled:
        return False
    
    try:
        # Run synthesis in thread to avoid blocking
        pcm_data = await asyncio.to_thread(synthesize_speech_sync, text)
        
        if not pcm_data:
            return False
        
        # Send audio in chunks to prevent timeout on large responses
        chunk_size = 48000  # ~1 second of audio at 24kHz, 16-bit
        chunks_sent = 0
        
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            await websocket.send_bytes(chunk)
            chunks_sent += 1
        
        logger.info(f"🔊 Sent {chunks_sent} audio chunks to client")
        return True
        
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        return False


def check_end_session(transcript: str) -> bool:
    """Check if transcript contains end session phrase."""
    transcript_lower = transcript.lower().strip()
    for phrase in END_SESSION_PHRASES:
        if phrase in transcript_lower:
            return True
    return False


def generate_response_sync(user_message: str, conversation_history: list, emotion: str | None = None) -> str:
    """
    Generate LLM response (synchronous).
    Called via asyncio.to_thread() to avoid blocking.
    
    Args:
        user_message: The transcribed text from the user
        conversation_history: Previous conversation turns
        emotion: Optional detected emotion from SenseVoice
    """
    if llm_model is None or llm_processor is None:
        return "I apologize, but my language model isn't loaded. I can hear you, but I can't formulate a proper response."
    
    logger.info("🧠 Generating LLM response...")
    start_time = time.time()
    
    try:
        # Build conversation for Gemma 3 chat format (multimodal structure)
        # System prompt as first user message with assistant acknowledgment
        messages = [
            {"role": "user", "content": [{"type": "text", "text": ICARUS_SYSTEM_PROMPT + "\n\nAcknowledge this persona briefly."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Understood. I'm Icarus, ready to assist with precision and perhaps a touch of wit. What do you need?"}]}
        ]
        
        # Add conversation history (convert to Gemma 3 format)
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })
        
        # Build user message with emotion context if available
        emotion_context = ""
        if emotion and emotion in EMOTION_MAP and EMOTION_MAP[emotion]:
            emotion_context = f"[The user sounds {EMOTION_MAP[emotion]}] "
        
        full_user_message = emotion_context + user_message
        
        # Add current user message in Gemma 3 format
        messages.append({"role": "user", "content": [{"type": "text", "text": full_user_message}]})
        
        # Apply chat template using processor
        inputs = llm_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(llm_model.device)
        
        input_len = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
            )
        
        # Decode only the new tokens
        response = llm_processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()
        
        elapsed = time.time() - start_time
        logger.info(f"🧠 LLM completed in {elapsed:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "I seem to have hit some turbulence. Could you repeat that?"


async def generate_response(user_message: str, conversation_history: list, emotion: str | None = None) -> str:
    """Async wrapper for LLM response generation."""
    return await asyncio.to_thread(generate_response_sync, user_message, conversation_history, emotion)


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
                            
                            # Transcribe with SenseVoice
                            session_state = SessionState.PROCESSING
                            transcript, emotion = await transcribe_audio(complete_audio)
                            logger.info(f"📝 Transcript: {transcript}")
                            
                            # Send CLEAN transcript to client (no emotion tags)
                            await websocket.send_text(f"TRANSCRIPT:{transcript}")
                            
                            # Check for end session command
                            if check_end_session(transcript):
                                logger.info("👋 End session command detected")
                                await websocket.send_text("STATE:IDLE")
                                session_state = SessionState.IDLE
                                conversation_history = []  # Clear history on session end
                            else:
                                # Generate LLM response (with emotion context for Gemma)
                                session_state = SessionState.GENERATING
                                llm_response = await generate_response(transcript, conversation_history, emotion)
                                logger.info(f"🤖 Response: {llm_response}")
                                
                                # Update conversation history (store clean text only)
                                conversation_history.append({"role": "user", "content": transcript})
                                conversation_history.append({"role": "assistant", "content": llm_response})
                                
                                # Trim history if too long
                                if len(conversation_history) > MAX_CONVERSATION_TURNS * 2:
                                    conversation_history = conversation_history[-MAX_CONVERSATION_TURNS * 2:]
                                
                                # Send response to client
                                await websocket.send_text(f"RESPONSE:{llm_response}")
                                
                                # Synthesize and stream audio
                                if tts_enabled:
                                    audio_sent = await synthesize_speech(llm_response, websocket)
                                    if audio_sent:
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
    
    # Configure logging before uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")