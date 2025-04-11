import asyncio
import json
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
from loguru import logger
from simulation_stage3 import SimulationParams, Simulation

app = FastAPI()

class WSMessage(BaseModel):
    event: str
    local_dispatch_count: int = 0
    active_count: int = 0
    city: str = ""
    county: str = ""
    timestamp: float

class APIParams(BaseModel):
    api_url: str = "http://localhost:5000"
    seed: str = "default"
    targetDispatches: int = 100
    maxActiveCalls: int = 15
    poll_interval: float = 0.3
    status_interval: float = 5
    debug: bool = False

class ConnectionManager:
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

# We hold a single global simulation instance; adjust as needed for concurrency.
sim_instance = None
sim_thread = None

def run_simulation_thread(params: APIParams):
    global sim_instance
    sim_params = SimulationParams(
        api_url=params.api_url,
        seed=params.seed,
        targetDispatches=params.targetDispatches,
        maxActiveCalls=params.maxActiveCalls,
        poll_interval=params.poll_interval,
        status_interval=params.status_interval,
        debug=params.debug
    )
    sim_instance = Simulation(sim_params)
    sim_instance.run_simulation(manager.broadcast)

@app.post("/simulate")
def simulate(params: APIParams, background_tasks: BackgroundTasks):
    global sim_thread
    if sim_thread and sim_thread.is_alive():
        return {"status": "already running"}
    background_tasks.add_task(run_simulation_thread, params)
    return {"status": "simulation started", "params": params.dict()}

@app.get("/simulate/status")
def simulation_status():
    global sim_instance
    if not sim_instance:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    return sim_instance.get_local_status()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected.")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
