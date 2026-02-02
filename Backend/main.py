from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging to track the "Split Brain" connection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IcarusBrain")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev, tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Status": "Icarus Brain Online"}

# This is the endpoint your client.py is trying to hit
@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via SSH Tunnel.")
    
    try:
        while True:
            # 1. Receive raw audio bytes from the "Ears" (Laptop)
            data = await websocket.receive_bytes()
            
            # 2. (Future) Feed 'data' into VAD/Whisper here
            # For now, just log the heartbeat so we know it works
            packet_size = len(data)
            
            # Reduce log noise: only print every 10th packet or if silence breaks
            logger.info(f"Processing audio packet: {packet_size} bytes")
            
            # 3. Send ACK (Optional, keeps the socket alive)
            await websocket.send_text("ACK")
            
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is crucial so it listens on the Tailscale/SSH interface, not just local loopback
    uvicorn.run(app, host="0.0.0.0", port=8000)