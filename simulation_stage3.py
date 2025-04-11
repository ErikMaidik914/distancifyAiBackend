import os
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

SESSION = None
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()

def pause_simulation():
    PAUSE_EVENT.clear()

def resume_simulation():
    PAUSE_EVENT.set()

def stop_simulation():
    STOP_EVENT.set()

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

def create_session(base_url, retries=3, backoff=0.5):
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

def _initialize_supply(session):
    # Example for a single endpoint
    # Adjust or add for multiple resource types if needed
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

def run_simulation(params):
    global SESSION
    SESSION = create_session(params.api_url)
    local_supply = {}
    tree = None
    supply_keys = []
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
        local_supply, tree_obj, supply_keys = _initialize_supply(SESSION)
    except Exception as e:
        logger.error(f"Supply initialization failed: {e}")
        return

    tree = tree_obj
    lock = threading.Lock()
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
        dist, idxs = tree.query(pt, k=len(supply_keys))
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
