import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

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

SESSION = None
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()
lock = threading.Lock()

def pause_simulation():
    PAUSE_EVENT.clear()

def resume_simulation():
    PAUSE_EVENT.set()

def stop_simulation():
    STOP_EVENT.set()

def create_session(base_url, retries=5, backoff=1):
    s = requests.Session()
    r = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504]
    )
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.base_url = base_url
    return s

def _build_city_kdtree(session):
    url = f"{session.base_url}/medical/search"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    points = []
    keys = []
    for d in data:
        lat = d.get("latitude", 0)
        lon = d.get("longitude", 0)
        if lat is None or lon is None:
            continue
        county = d.get("county", "").strip()
        city = d.get("city", "").strip()
        if not county or not city:
            continue
        points.append((lat, lon))
        keys.append((county, city, lat, lon))
    tree = KDTree(points) if points else None
    return tree, keys

def _fetch_current_supply(session, county, city):
    url = f"{session.base_url}/medical/searchbycity?county={county}&city={city}"
    try:
        resp = session.get(url)
        if resp.ok:
            items = resp.json()
            if items and isinstance(items, list) and len(items) > 0:
                qty = items[0].get("quantity", 0)
                if qty is None or qty < 0:
                    return 0
                return qty
        return 0
    except Exception:
        return 0

def _dispatch(session, srcCounty, srcCity, tgtCounty, tgtCity, qty):
    url = f"{session.base_url}/medical/dispatch"
    resp = session.post(url, json={
        "sourceCounty": srcCounty,
        "sourceCity": srcCity,
        "targetCounty": tgtCounty,
        "targetCity": tgtCity,
        "quantity": qty
    })
    return resp.ok

def _get_next_emergency(session):
    url = f"{session.base_url}/calls/next"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

def _get_status(session):
    url = f"{session.base_url}/control/status"
    resp = session.get(url)
    return resp.json() if resp.ok else None

def get_status():
    if SESSION is None:
        return None
    return _get_status(SESSION)

def run_simulation(params: APIParams):
    global SESSION
    SESSION = create_session(params.api_url)
    active_count = 0
    local_dispatch_count = 0

    reset_url = (
        f"{SESSION.base_url}/control/reset"
        f"?seed={params.seed}"
        f"&targetDispatches={params.targetDispatches}"
        f"&maxActiveCalls={params.maxActiveCalls}"
    )
    r = SESSION.post(reset_url)
    if not r.ok:
        logger.error(f"Reset failed: {r.status_code} {r.text}")
        return

    try:
        city_tree, city_keys = _build_city_kdtree(SESSION)
    except Exception as e:
        logger.error(f"Building KDTree failed: {e}")
        return

    pool = ThreadPoolExecutor(max_workers=params.maxActiveCalls)
    futures = []
    last_status_check = time.time()

    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        needed = sum(x["Quantity"] for x in call.get("requests", []))
        if needed <= 0:
            with lock:
                active_count -= 1
            return

        pt = (call["latitude"], call["longitude"])
        if not city_tree or not city_keys:
            with lock:
                active_count -= 1
            return

        dist, idxs = city_tree.query(pt, k=len(city_keys))
        remain = needed

        with lock:
            for i in idxs if hasattr(idxs, "__iter__") else [idxs]:
                county, city, lat, lon = city_keys[i]
                current_supply = _fetch_current_supply(SESSION, county, city)
                if current_supply <= 0:
                    continue
                use = min(current_supply, remain)
                ok = _dispatch(SESSION, county, city, call["county"], call["city"], use)
                if ok:
                    remain -= use
                    local_dispatch_count += use
                if remain <= 0:
                    break
            active_count -= 1

    while True:
        if STOP_EVENT.is_set():
            break
        PAUSE_EVENT.wait()
        if local_dispatch_count >= params.targetDispatches:
            break
        if time.time() - last_status_check >= params.status_interval:
            st = _get_status(SESSION)
            if st and st.get("totalDispatches", 0) >= params.targetDispatches:
                break
            last_status_check = time.time()

        with lock:
            curr = active_count
        if curr < params.maxActiveCalls:
            try:
                call = _get_next_emergency(SESSION)
            except Exception as e:
                logger.error(f"Error fetching call: {e}")
                time.sleep(params.poll_interval)
                continue
            if call:
                with lock:
                    active_count += 1
                futures.append(pool.submit(process_emergency, call))
            else:
                time.sleep(params.poll_interval)
        else:
            time.sleep(params.poll_interval)

        futures = [f for f in futures if not f.done()]

    stop_resp = SESSION.post(f"{SESSION.base_url}/control/stop")
    if not stop_resp.ok:
        logger.error(f"Stop failed: {stop_resp.status_code} {stop_resp.text}")

    STOP_EVENT.clear()

def start_simulation_thread(params: APIParams):
    thread = threading.Thread(target=run_simulation, args=(params,), daemon=True)
    thread.start()

@app.get("/health")
def health_check():
    return {"status": "ok"}

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
