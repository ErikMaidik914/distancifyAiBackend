import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger
from simulation import SimulationParams, run_simulation, get_status, pause_simulation, resume_simulation, stop_simulation

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class APIParams(BaseModel):
    api_url: str = ""
    seed: str = "default"
    targetDispatches: int = 100
    maxActiveCalls: int = 15
    poll_interval: float = 0.3
    status_interval: float = 5

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

@app.post("/simulate")
def simulate(params: APIParams, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_simulation_thread, params)
    return {"status": "simulation started", "params": params.dict()}

@app.post("/simulate/pause")
def pause():
    pause_simulation()
    return {"status": "simulation paused"}

@app.post("/simulate/resume")
def resume():
    resume_simulation()
    return {"status": "simulation resumed"}

@app.post("/simulate/stop")
def stop():
    stop_simulation()
    return {"status": "simulation stopped"}

@app.get("/simulate/status")
def simulation_status():
    s = get_status()
    if s is None:
        raise HTTPException(status_code=500, detail="Error fetching simulation status")
    return s

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
