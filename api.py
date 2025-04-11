import asyncio
import json
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
from simulation import SimulationParams, run_simulation, get_status
from loguru import logger

app = FastAPI()

class WSMessage(BaseModel):
    event: str
    local_dispatch_count: int = 0
    active_count: int = 0
    city: str = ""
    county: str = ""
    timestamp: float

class APIParams(BaseModel):
    seed: str = "default"
    targetDispatches: int = 100
    maxActiveCalls: int = 15
    poll_interval: float = 0.3
    status_interval: float = 5

class ConnectionManager:
    """
    Manages active websocket connections.
    """
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def start_simulation_thread(params: APIParams):
    """
    Starts the simulation on a separate thread.
    """
    sim_params = SimulationParams(
        seed=params.seed,
        targetDispatches=params.targetDispatches,
        maxActiveCalls=params.maxActiveCalls,
        poll_interval=params.poll_interval,
        status_interval=params.status_interval
    )
    thread = threading.Thread(target=run_simulation, args=(sim_params, manager.broadcast))
    thread.start()

@app.post("/simulate")
def simulate(params: APIParams, background_tasks: BackgroundTasks):
    """
    Starts the simulation and returns immediately.
    """
    background_tasks.add_task(start_simulation_thread, params)
    return {"status": "simulation started", "params": params.dict()}

@app.get("/simulate/status")
def simulation_status():
    """
    Returns the current simulation status.
    """
    status = get_status()
    if status is None:
        raise HTTPException(status_code=500, detail="Error fetching simulation status")
    return status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles incoming websocket connections and maintains updates.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive; optionally process incoming messages if needed.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected.")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
