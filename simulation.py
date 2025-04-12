import os
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree

# Global session and control flags used for simulation state management
SESSION = None
PAUSE_EVENT = threading.Event()  # Flag to pause/resume simulation
PAUSE_EVENT.set()                # Set by default (running state)
STOP_EVENT = threading.Event()   # Flag to stop simulation

# Control function to pause the simulation
def pause_simulation():
    PAUSE_EVENT.clear()

# Control function to resume the simulation
def resume_simulation():
    PAUSE_EVENT.set()

# Control function to stop the simulation
def stop_simulation():
    STOP_EVENT.set()

#
#
# Class holding the simulation configuration parameters
class SimulationParams:
    def __init__(
        self,
        api_url=None,
        seed="default",
        targetDispatches=100,
        maxActiveCalls=15,
        poll_interval=0.3,
        status_interval=5
    ):
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval

# Create a resilient HTTP session with retry logic
def create_session(base_url, retries=3, backoff=0.5):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.base_url = base_url
    return s

# Fetch and index supply locations into a KDTree for fast nearest-neighbor search
def _initialize_supply(session):
    supply_data = {}
    url = f"{session.base_url}/medical/search"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    points, keys = [], []
    for d in data:
        k = (d["county"], d["city"])
        supply_data[k] = {
            "quantity": d["quantity"],
            "lat": d["latitude"],
            "lon": d["longitude"]
        }
        keys.append(k)
        points.append((d["latitude"], d["longitude"]))
    return supply_data, KDTree(points), keys

# Send a dispatch request to the backend API
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

# Fetch the next emergency call from the API
def _get_next_emergency(session):
    url = f"{session.base_url}/calls/next"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

# Fetch the current global simulation status
def _get_status(session):
    url = f"{session.base_url}/control/status"
    resp = session.get(url)
    return resp.json() if resp.ok else None

# Public getter for external status polling (used by API layer)
def get_status():
    if SESSION is None:
        return None
    return _get_status(SESSION)

# Main simulation entry point
def run_simulation(params):
    global SESSION
    SESSION = create_session(params.api_url)

    # Simulation state
    local_supply = {}
    tree = None
    supply_keys = []
    active_count = 0
    local_dispatch_count = 0

    # Reset the backend simulation state
    reset_url = f"{SESSION.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = SESSION.post(reset_url)
    if not r.ok:
        logger.error(f"Reset call failed: {r.status_code} {r.text}")
        return

    # Load supply data and build KDTree
    try:
        local_supply, tree_obj, supply_keys = _initialize_supply(SESSION)
    except Exception as e:
        logger.error(f"Supply initialization failed: {e}")
        return
    tree = tree_obj
    lock = threading.Lock()  # Used to safely update shared state across threads

    # Handle a single emergency call in a separate thread
    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        needed = sum(x["Quantity"] for x in call.get("requests", []))
        if needed <= 0:
            with lock:
                active_count -= 1
            return

        pt = (call["latitude"], call["longitude"])
        dist, idxs = tree.query(pt, k=len(supply_keys))  # Get nearest supply points
        remain = needed
        with lock:
            for i in idxs:
                key = supply_keys[i]
                av = local_supply[key]["quantity"]
                if av <= 0:
                    continue
                use = min(av, remain)
                ok = _dispatch(SESSION, key[0], key[1], call["county"], call["city"], use)
                if ok:
                    local_supply[key]["quantity"] -= use
                    remain -= use
                    local_dispatch_count += use
                if remain <= 0:
                    break
            active_count -= 1

    threads = []
    last_status_check = time.time()

    # Main event loop
    while True:
        if STOP_EVENT.is_set():  # Exit if stop was triggered
            break
        PAUSE_EVENT.wait()       # Wait here if paused
        if local_dispatch_count >= params.targetDispatches:
            break

        # Periodically check simulation status from backend
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
                t = threading.Thread(target=process_emergency, args=(call,))
                t.start()
                threads.append(t)
            else:
                time.sleep(params.poll_interval)
        else:
            time.sleep(params.poll_interval)

        # Remove finished threads from tracking list
        threads = [t for t in threads if t.is_alive()]

    # Notify backend that simulation has ended
    stop = SESSION.post(f"{SESSION.base_url}/control/stop")
    if not stop.ok:
        logger.error(f"Stop call failed: {stop.status_code} {stop.text}")

    # Wait for all emergency threads to finish
    for t in threads:
        t.join()

    STOP_EVENT.clear()  # Reset stop flag for future runs
