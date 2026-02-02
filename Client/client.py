# client.py
import asyncio
import pyaudio
import websockets
import logging

# Audio Configuration (Standard for Whisper/VAD)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096  # 256ms of audio per packet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IcarusClient")

async def send_audio():
    # Connect to LOCALHOST because of the SSH tunnel
    uri = "ws://localhost:8000/ws/audio"
    
    p = pyaudio.PyAudio()
    
    # Open the microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    logger.info(f"Connecting to Icarus Brain at {uri}...")

    async with websockets.connect(uri) as websocket:
        logger.info("Connected. Streaming audio...")
        
        try:
            while True:
                # Read raw bytes from hardware
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Send to server
                await websocket.send(data)
                
                # Optional: Read response (e.g., "Silence" or "Speech")
                response = await websocket.recv()
                # print(f"Server says: {response}") 
                
        except KeyboardInterrupt:
            logger.info("Stopping stream...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio())
    except Exception as e:
        logger.error(f"Connection failed. Is the SSH tunnel active? Error: {e}")