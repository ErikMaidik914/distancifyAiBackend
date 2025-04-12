import os
import sys
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

# Enable debug logging to stdout
logger.add(sys.stdout, level="DEBUG")

# Global simulation state variables
SESSION = None
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()

# Control functions for pausing, resuming, and stopping the simulation
def pause_simulation():
    PAUSE_EVENT.clear()

def resume_simulation():
    PAUSE_EVENT.set()

def stop_simulation():
    STOP_EVENT.set()

# Simulation configuration wrapper
class SimulationParams:
    def __init__(self, api_url=None, seed="default", targetDispatches=100, maxActiveCalls=15, poll_interval=0.3, status_interval=5):
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval

# HTTP session with retry capabilities
def create_session(base_url, retries=3, backoff=0.5):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.base_url = base_url
    return s

# Wrapper for GET requests with error handling and timeout
def safe_get(session, url, timeout=5):
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.error(f"GET {url} failed: {e}")
        return None

# Wrapper for POST requests with error handling and timeout
def safe_post(session, url, json=None, timeout=5):
    try:
        resp = session.post(url, json=json, timeout=timeout)
        return resp
    except Exception as e:
        logger.error(f"POST {url} failed: {e}")
        return None

# Fetch and build initial KDTree of all supply points
def _initialize_supply(session):
    supply_data = {}
    url = f"{session.base_url}/medical/search"
    resp = safe_get(session, url)
    if not resp:
        raise Exception("Supply initialization failed")
    data = resp.json()
    points, keys = [], []
    for d in data:
        if d.get("quantity", -1) < 0 or d.get("latitude") is None or d.get("longitude") is None:
            continue
        k = (d["county"], d["city"])
        supply_data[k] = {"quantity": d["quantity"], "lat": d["latitude"], "lon": d["longitude"]}
        keys.append(k)
        points.append((d["latitude"], d["longitude"]))
    return supply_data, KDTree(points), keys

# Refresh real-time quantity of supply for a specific city
def refresh_supply(session, county, city):
    url = f"{session.base_url}/medical/searchbycity?county={county}&city={city}"
    resp = safe_get(session, url)
    if not resp:
        raise Exception("Refresh supply failed")
    data = resp.json()
    if isinstance(data, int):
        return {"quantity": data}
    if data is None or data.get("quantity", -1) < 0:
        raise Exception("Bad data from refresh supply")
    return data

# Dispatch supply from source to target
def _dispatch(session, srcCounty, srcCity, tgtCounty, tgtCity, qty):
    url = f"{session.base_url}/medical/dispatch"
    resp = safe_post(session, url, json={
        "sourceCounty": srcCounty,
        "sourceCity": srcCity,
        "targetCounty": tgtCounty,
        "targetCity": tgtCity,
        "quantity": qty
    })
    if not resp or not resp.ok:
        logger.error(f"Dispatch from {srcCounty}-{srcCity} to {tgtCounty}-{tgtCity} failed")
        return False
    return True

# Fetch the next emergency call from the backend
def _get_next_emergency(session):
    url = f"{session.base_url}/calls/next"
    resp = safe_get(session, url)
    if not resp:
        return None
    if resp.status_code == 404:
        return None
    data = resp.json()
    required = ["latitude", "longitude", "county", "city", "requests"]
    if not all(k in data for k in required):
        raise Exception("Bad emergency call data")
    return data

# Get simulation status from backend
def _get_status(session):
    url = f"{session.base_url}/control/status"
    resp = safe_get(session, url)
    return resp.json() if resp and resp.ok else None

# Public status endpoint
def get_status():
    if SESSION is None:
        return None
    return _get_status(SESSION)

# Main simulation execution logic
def run_simulation(params):
    global SESSION
    SESSION = create_session(params.api_url)
    local_supply = {}
    tree = None
    supply_keys = []
    active_count = 0
    local_dispatch_count = 0

    # Reset backend simulation state
    reset_url = f"{SESSION.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = safe_post(SESSION, reset_url)
    if not r or not r.ok:
        logger.error(f"Reset failed: {r.status_code if r else 'No response'}")
        return

    # Build spatial tree for all available supply points
    try:
        local_supply, tree_obj, supply_keys = _initialize_supply(SESSION)
    except Exception as e:
        logger.error(f"Supply initialization failed: {e}")
        return

    tree = tree_obj
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=params.maxActiveCalls)
    futures = []
    last_status_check = time.time()

    # Thread worker to handle an individual emergency call
    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        needed = sum(x["Quantity"] for x in call.get("requests", []))
        if needed <= 0:
            with lock:
                active_count -= 1
            return
        pt = (call["latitude"], call["longitude"])
        _, idxs = tree.query(pt, k=len(supply_keys))
        remain = needed
        with lock:
            for i in idxs:
                key = supply_keys[i]
                try:
                    updated = refresh_supply(SESSION, key[0], key[1])
                    local_supply[key]["quantity"] = updated["quantity"]
                except Exception as e:
                    logger.error(f"Failed to refresh supply for {key}: {e}")
                    continue
                av = local_supply[key]["quantity"]
                if av <= 0:
                    continue
                use = min(av, remain)
                if use <= 0:
                    continue
                if _dispatch(SESSION, key[0], key[1], call["county"], call["city"], use):
                    local_supply[key]["quantity"] -= use
                    remain -= use
                    local_dispatch_count += use
                    logger.debug(f"Dispatched {use} from {key} to {call['county']}-{call['city']} (Remaining need: {remain})")
                else:
                    logger.error(f"Dispatch failed for {key} to {call['county']}-{call['city']}")
                if remain <= 0:
                    break
            if remain > 0:
                logger.warning(f"Emergency {call['county']}-{call['city']} not fully served. Remaining: {remain}")
            active_count -= 1

    # Continuous loop for fetching and processing emergencies
    while True:
        if STOP_EVENT.is_set():
            break
        PAUSE_EVENT.wait()

        # Local dispatch target reached
        if local_dispatch_count >= params.targetDispatches:
            break

        # Periodically check backend for target fulfillment
        if time.time() - last_status_check >= params.status_interval:
            st = _get_status(SESSION)
            if st and st.get("totalDispatches", 0) >= params.targetDispatches:
                break
            last_status_check = time.time()

        # If we can handle more concurrent emergencies
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

        # Clean up completed futures
        futures = [f for f in futures if not f.done()]

    # Stop simulation on backend
    stop_resp = safe_post(SESSION, f"{SESSION.base_url}/control/stop")
    if not stop_resp or not stop_resp.ok:
        logger.error(f"Stop failed: {stop_resp.status_code if stop_resp else 'No response'}")
    STOP_EVENT.clear()

# Entry point for running this script independently
if __name__ == '__main__':
    params = SimulationParams(api_url="http://localhost:5000", seed="test", targetDispatches=10, maxActiveCalls=5)
    run_simulation(params)
