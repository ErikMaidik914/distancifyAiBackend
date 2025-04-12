import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
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
    global simulation_instance
    if simulation_instance is None or not simulation_instance.thread.is_alive():
        sp = SimulationParams(api_url=params.api_url, seed=params.seed, targetDispatches=params.targetDispatches, maxActiveCalls=params.maxActiveCalls, poll_interval=params.poll_interval, status_interval=params.status_interval, emergency_types=params.emergency_types, debug_mode=params.debug_mode)
        simulation_instance = Simulation(sp)
        simulation_instance.start()

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
    if simulation_instance and simulation_instance.thread.is_alive():
        simulation_instance.pause()
        return {"status": "simulation paused"}
    raise HTTPException(status_code=400, detail="Simulation not running")

# Resume a paused simulation
@app.post("/simulate/resume")
def resume():
    if simulation_instance and simulation_instance.thread.is_alive():
        simulation_instance.resume()
        return {"status": "simulation resumed"}
    raise HTTPException(status_code=400, detail="Simulation not running")

# Force stop the simulation
@app.post("/simulate/stop")
def stop():
    if simulation_instance and simulation_instance.thread.is_alive():
        simulation_instance.stop()
        return {"status": "simulation stop requested"}
    raise HTTPException(status_code=400, detail="Simulation not running")

# Retrieve the current status of the simulation
@app.get("/simulate/status")
def simulation_status(api_url: str = None):
    base_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
    session = create_session(base_url)
    s = get_status(session)
    if s is None:
        raise HTTPException(status_code=500, detail="Error fetching simulation status")
    return s

# Run the FastAPI server when script is launched directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
