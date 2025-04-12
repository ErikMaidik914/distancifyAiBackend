import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Import simulation logic and control functions
from simulation import SimulationParams, run_simulation, get_status, pause_simulation, resume_simulation, stop_simulation

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins and methods to support frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define the expected structure for API parameters
class APIParams(BaseModel):
    api_url: str = ""               # Base URL of the simulation backend
    seed: str = "default"           # Seed used to reset and randomize the simulation
    targetDispatches: int = 100     # Total number of dispatches to perform before stopping
    maxActiveCalls: int = 15        # Maximum number of simultaneous active emergency calls
    poll_interval: float = 0.3      # Time interval between checking for new emergencies
    status_interval: float = 5      # Time interval for polling the simulation status

# Function that starts the simulation in a background thread
def start_simulation_thread(params: APIParams):
    # Convert API params into simulation-specific params
    sim_params = SimulationParams(
        api_url=params.api_url,
        seed=params.seed,
        targetDispatches=params.targetDispatches,
        maxActiveCalls=params.maxActiveCalls,
        poll_interval=params.poll_interval,
        status_interval=params.status_interval
    )
    # Run simulation in a daemon thread to avoid blocking the main thread
    thread = threading.Thread(target=run_simulation, args=(sim_params,), daemon=True)
    thread.start()

# Endpoint to start the simulation
@app.post("/simulate")
def simulate(params: APIParams, background_tasks: BackgroundTasks):
    # Queue the simulation to run in the background
    background_tasks.add_task(start_simulation_thread, params)
    return {"status": "simulation started", "params": params.dict()}

# Endpoint to pause the simulation
@app.post("/simulate/pause")
def pause():
    pause_simulation()
    return {"status": "simulation paused"}

# Endpoint to resume the paused simulation
@app.post("/simulate/resume")
def resume():
    resume_simulation()
    return {"status": "simulation resumed"}

# Endpoint to stop the simulation entirely
@app.post("/simulate/stop")
def stop():
    stop_simulation()
    return {"status": "simulation stopped"}

# Endpoint to get the current status of the simulation
@app.get("/simulate/status")
def simulation_status():
    s = get_status()
    if s is None:
        raise HTTPException(status_code=500, detail="Error fetching simulation status")
    return s

# Run the FastAPI server when this script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
