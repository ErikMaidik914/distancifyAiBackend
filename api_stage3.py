import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Import stage 3 simulation logic and control functions
from simulation_stage3 import (
    SimulationParams,
    run_simulation,
    get_status,
    pause_simulation,
    resume_simulation,
    stop_simulation,
)

# Initialize FastAPI app instance
app = FastAPI()

# Enable CORS for all origins to allow frontend communication (development-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define expected input schema for /simulate request
class APIParams(BaseModel):
    api_url: str = ""               # Base API endpoint for the local simulation
    seed: str = "default"           # Simulation seed for deterministic results
    targetDispatches: int = 100     # Maximum number of dispatches to perform
    maxActiveCalls: int = 15        # Limit of concurrent emergency call processing
    poll_interval: float = 0.3      # Delay between polling for new emergencies
    status_interval: float = 5      # Frequency for checking simulation status

# Function to spawn a simulation thread without blocking the main server
def start_simulation_thread(params: APIParams):
    sim_params = SimulationParams(
        api_url=params.api_url,
        seed=params.seed,
        targetDispatches=params.targetDispatches,
        maxActiveCalls=params.maxActiveCalls,
        poll_interval=params.poll_interval,
        status_interval=params.status_interval
    )
    thread = threading.Thread(target=run_simulation, args=(sim_params,), daemon=True)
    thread.start()

# Basic health check endpoint for monitoring
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Start simulation with provided parameters
@app.post("/simulate")
def simulate(params: APIParams, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_simulation_thread, params)
    return {"status": "simulation started", "params": params.dict()}

# Pause simulation execution
@app.post("/simulate/pause")
def pause():
    pause_simulation()
    return {"status": "simulation paused"}

# Resume a paused simulation
@app.post("/simulate/resume")
def resume():
    resume_simulation()
    return {"status": "simulation resumed"}

# Force stop the simulation
@app.post("/simulate/stop")
def stop():
    stop_simulation()
    return {"status": "simulation stopped"}

# Retrieve the current status of the simulation
@app.get("/simulate/status")
def simulation_status():
    s = get_status()
    if s is None:
        raise HTTPException(status_code=500, detail="Error fetching simulation status")
    return s

# Run the FastAPI server when script is launched directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
