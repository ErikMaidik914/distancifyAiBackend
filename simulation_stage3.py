import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree

# Global control events for simulation flow
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()            # Initially allow simulation to run
STOP_EVENT = threading.Event()  # Used to interrupt and stop the simulation

# Class that holds configuration for a simulation run
class SimulationParams:
    def __init__(self, api_url=None, seed="default", targetDispatches=100, maxActiveCalls=15,
                 poll_interval=0.3, status_interval=5, emergency_types=None, debug_mode=False):
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.emergency_types = emergency_types or ["Medical", "Fire", "Police", "Rescue", "Utility"]
        self.debug_mode = debug_mode

# Create a session with retry logic for resilience
def create_session(base_url, retries=3, backoff=0.5):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=r)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.base_url = base_url
    return s

# Load initial supply data for each emergency type and index with KDTree
def initialize_supply(session, emergency_types):
    supplies = {}
    for etype in emergency_types:
        url = f"{session.base_url}/{etype.lower()}/search"
        resp = session.get(url)
        if resp.ok:
            data = resp.json()
            local_supply = {}
            supply_keys = []
            supply_points = []
            for entry in data:
                key = (entry["county"], entry["city"])
                local_supply[key] = {
                    "quantity": entry["quantity"],
                    "lat": entry["latitude"],
                    "lon": entry["longitude"]
                }
                supply_keys.append(key)
                supply_points.append((entry["latitude"], entry["longitude"]))
            tree = KDTree(supply_points) if supply_points else None
            supplies[etype] = {
                "local_supply": local_supply,
                "supply_keys": supply_keys,
                "supply_points": supply_points,
                "tree": tree
            }
            logger.info("{} supply loaded.", etype)
        else:
            logger.error("{} supply loading failed: {} {}", etype, resp.status_code, resp.text)
    return supplies

# Send a dispatch request to the simulation backend
def dispatch(session, etype, srcCounty, srcCity, tgtCounty, tgtCity, qty, debug_mode=False):
    url = f"{session.base_url}/{etype.lower()}/dispatch"
    payload = {
        "sourceCounty": srcCounty,
        "sourceCity": srcCity,
        "targetCounty": tgtCounty,
        "targetCity": tgtCity,
        "quantity": qty
    }
    resp = session.post(url, json=payload)
    if resp.ok:
        if debug_mode:
            logger.debug("Dispatched {} {} from {} {} to {} {}.", qty, etype, srcCity, srcCounty, tgtCity, tgtCounty)
        return True
    logger.error("Dispatch failed for {} from {} {} to {} {}: {} {}", etype, srcCity, srcCounty, tgtCity, tgtCounty, resp.status_code, resp.text)
    return False

# Get the next emergency call from the API
def get_next_emergency(session):
    url = f"{session.base_url}/calls/next"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    if resp.ok:
        try:
            return resp.json()
        except Exception as e:
            logger.error("JSON parse error: {} {}", e, resp.text)
    else:
        logger.error("Error calling /calls/next: {} {}", resp.status_code, resp.text)
    return None

# Get current global simulation status from the API
def get_status(session):
    url = f"{session.base_url}/control/status"
    resp = session.get(url)
    if resp.ok:
        return resp.json()
    logger.error("Error fetching status: {} {}", resp.status_code, resp.text)
    return None

# Main simulation loop
def run_simulation(params):
    session = create_session(params.api_url)

    # Reset backend simulation with configuration
    reset_url = f"{session.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = session.post(reset_url)
    if not r.ok:
        logger.error("Reset failed: {} {}", r.status_code, r.text)
        return
    logger.info("Simulation reset: {}", r.json())

    # Load supply for all emergency types and initialize KDTree
    supplies = initialize_supply(session, params.emergency_types)

    # Shared state and thread pool
    active_lock = threading.Lock()
    supply_lock = threading.Lock()
    active_count = 0
    local_dispatch_count = 0
    pool = ThreadPoolExecutor(max_workers=params.maxActiveCalls)
    futures = []
    last_status_check = time.time()

    # Function to handle a single emergency request
    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        for req in call.get("requests", []):
            needed = req["Quantity"]
            if needed <= 0:
                logger.debug("No {} units required at {} {}.", req["Type"], call["city"], call["county"])
                continue
            supply_data = supplies.get(req["Type"])
            if not supply_data or not supply_data["tree"]:
                logger.error("No supply available for {}.", req["Type"])
                continue
            pt = (call["latitude"], call["longitude"])
            distances, indices = supply_data["tree"].query(pt, k=len(supply_data["supply_points"]))
            remaining = needed
            with supply_lock:
                for idx in indices:
                    key = supply_data["supply_keys"][idx]
                    available = supply_data["local_supply"][key]["quantity"]
                    if available <= 0:
                        continue
                    use = min(available, remaining)
                    if dispatch(session, req["Type"], key[0], key[1], call["county"], call["city"], use, params.debug_mode):
                        supply_data["local_supply"][key]["quantity"] -= use
                        remaining -= use
                        local_dispatch_count += use
                        if params.debug_mode:
                            logger.debug("Dispatch count updated: {}", local_dispatch_count)
                        if remaining <= 0:
                            break
                    else:
                        logger.error("Dispatch error for {} at {} {}.", req["Type"], call["city"], call["county"])
            if remaining > 0:
                logger.warning("Not fully dispatched for {} at {} {}; missing {} units.", req["Type"], call["city"], call["county"], remaining)
        with active_lock:
            active_count -= 1
            logger.info("Processed emergency at {} {}. Active: {}", call["city"], call["county"], active_count)

    # Main dispatching loop
    while True:
        if STOP_EVENT.is_set():
            break
        PAUSE_EVENT.wait()
        if local_dispatch_count >= params.targetDispatches:
            logger.info("Local target reached: {}.", local_dispatch_count)
            break
        if time.time() - last_status_check >= params.status_interval:
            status = get_status(session)
            if status and status.get("totalDispatches", 0) >= params.targetDispatches:
                logger.info("Remote target reached.")
                break
            last_status_check = time.time()
        with active_lock:
            current = active_count
        if current < params.maxActiveCalls:
            emergency = get_next_emergency(session)
            if emergency:
                with active_lock:
                    active_count += 1
                futures.append(pool.submit(process_emergency, emergency))
                logger.info("Submitted emergency at {} {}. Active: {}", emergency["city"], emergency["county"], active_count)
            else:
                time.sleep(params.poll_interval)
        else:
            time.sleep(params.poll_interval)
        futures = [f for f in futures if not f.done()]

    # Stop simulation on backend
    stop_resp = session.post(f"{session.base_url}/control/stop")
    if stop_resp.ok:
        logger.info("Simulation stopped: {}", stop_resp.json())
    else:
        logger.error("Stop failed: {} {}", stop_resp.status_code, stop_resp.text)

    # Reset stop flag for future runs
    STOP_EVENT.clear()

# Run simulation if the script is executed directly
if __name__ == "__main__":
    params = SimulationParams(seed="mySeed", targetDispatches=100, maxActiveCalls=15)
    run_simulation(params)
