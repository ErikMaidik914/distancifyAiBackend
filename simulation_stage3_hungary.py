import os
import time
import threading
import requests
import signal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from math import sqrt
from typing import Any, Dict, Optional, List, Tuple

# Global control events for simulation flow
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()  # Initially allow simulation to run
STOP_EVENT = threading.Event()  # Used to interrupt and stop the simulation

class SimulationParams:
    def __init__(self, api_url: Optional[str] = None, seed: str = "default", targetDispatches: int = 100,
                 maxActiveCalls: int = 15, poll_interval: float = 0.3, status_interval: float = 5,
                 emergency_types: Optional[list] = None, debug_mode: bool = False) -> None:
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.emergency_types = emergency_types or ["Medical", "Fire", "Police", "Rescue", "Utility"]
        self.debug_mode = debug_mode

def create_session(base_url: str, retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry_strategy = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.base_url = base_url  # type: ignore
    s.headers.update({"Content-Type": "application/json"})
    return s

def request_with_retry(session: requests.Session, method: str, url: str, max_retries: int = 3,
                         backoff: float = 0.5, **kwargs) -> Optional[requests.Response]:
    for attempt in range(max_retries):
        try:
            response = session.request(method, url, **kwargs)
            # For the /calls/next endpoint, a 404 means "end" so return immediately.
            if "/calls/next" in url and response.status_code == 404:
                return response
            if response.ok:
                return response
            if response.text and "Not started" in response.text:
                logger.error("Request {} {} returned 'Not started'.", method, url)
                return None
            logger.error("Request {} {} failed attempt {}: {} {}", method, url, attempt + 1,
                         response.status_code, response.text)
        except Exception as e:
            logger.exception("Request {} {} error attempt {}: {}", method, url, attempt + 1, e)
        time.sleep(backoff * (2 ** attempt))
    return None

def initialize_supply(session: requests.Session, emergency_types: List[str]) -> Dict[str, Any]:
    supplies = {}
    for etype in emergency_types:
        url = f"{session.base_url}/{etype.lower()}/search"
        resp = session.get(url)
        if resp.ok:
            try:
                data = resp.json()
            except Exception as e:
                logger.exception("Failed to parse JSON for {} supply: {}", etype, e)
                continue
            # Using a dict keyed by (county, city)
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

def dispatch(session: requests.Session, etype: str, srcCounty: str, srcCity: str,
             tgtCounty: str, tgtCity: str, qty: int, debug_mode: bool = False) -> bool:
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
    logger.error("Dispatch failed for {} from {} {} to {} {}: {} {}", etype, srcCity, srcCounty, tgtCity, tgtCounty,
                 resp.status_code, resp.text)
    return False

def get_next_emergency(session: requests.Session) -> Optional[Dict[str, Any]]:
    url = f"{session.base_url}/calls/next"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    if resp.ok:
        try:
            return resp.json()
        except Exception as e:
            logger.error("JSON parse error in get_next_emergency: {} {}", e, resp.text)
    else:
        logger.error("Error calling /calls/next: {} {}", resp.status_code, resp.text)
    return None

def get_status(session: requests.Session) -> Optional[Dict[str, Any]]:
    url = f"{session.base_url}/control/status"
    resp = session.get(url)
    if resp.ok:
        try:
            return resp.json()
        except Exception as e:
            logger.error("Error parsing JSON in get_status: {} {}", e, resp.text)
            return None
    logger.error("Error fetching status: {} {}", resp.status_code, resp.text)
    return None

def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def process_batch(session: requests.Session, batch: List[Dict[str, Any]], supplies: Dict[str, Any],
                  params: SimulationParams, supply_lock: threading.Lock) -> int:
    """
    Process a batch of emergency calls via global assignment for each emergency type.
    For each type, replicate calls and supply units according to needed and available quantities,
    build a cost matrix based on Euclidean distance and use the Hungarian algorithm.
    Dispatch each assigned unit.
    
    Returns the total number of units dispatched.
    """
    total_dispatched = 0

    for etype in params.emergency_types:
        # Extract emergency requests of this type from the batch.
        # For each call, there may be multiple requests; here we combine all requests for etype.
        calls_for_type = []
        # We'll keep a mapping for each call: index -> [ (call, remaining) ]
        for call in batch:
            for req in call.get("requests", []):
                if req["Type"] == etype and req["Quantity"] > 0:
                    # We'll record the call's location and needed units.
                    calls_for_type.append({"call": call, "needed": req["Quantity"]})
                    break  # Assume one request per type per call

        if not calls_for_type:
            continue

        # Replicate calls: each unit needed gets one row in cost matrix.
        rep_calls = []
        call_indices = []  # To later map back to the original call in calls_for_type
        for idx, item in enumerate(calls_for_type):
            needed = item["needed"]
            loc = (item["call"]["latitude"], item["call"]["longitude"])
            for _ in range(needed):
                rep_calls.append(loc)
                call_indices.append(idx)
        if not rep_calls:
            continue

        # Replicate supply units: each available unit gets one column.
        rep_supplies = []
        supply_keys = []
        supply_data = supplies[etype]["local_supply"]
        for key, details in supply_data.items():
            available = details["quantity"]
            loc = (details["lat"], details["lon"])
            for _ in range(available):
                rep_supplies.append(loc)
                supply_keys.append(key)
        if not rep_supplies:
            logger.warning("No available supply for {} in this batch.", etype)
            continue

        # Build cost matrix.
        cost_matrix = np.zeros((len(rep_calls), len(rep_supplies)))
        for i, call_loc in enumerate(rep_calls):
            for j, supp_loc in enumerate(rep_supplies):
                cost_matrix[i, j] = euclidean_distance(call_loc, supp_loc)

        # Solve assignment problem.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # For each assignment, dispatch one unit.
        # Use supply_lock to update supplies consistently.
        with supply_lock:
            for i, j in zip(row_ind, col_ind):
                # Determine which call and supply this represents.
                call_idx = call_indices[i]
                assigned_call = calls_for_type[call_idx]["call"]
                supply_key = supply_keys[j]
                # Dispatch one unit from supply_key to the call.
                if dispatch(session, etype, supply_key[0], supply_key[1],
                            assigned_call["county"], assigned_call["city"], 1, params.debug_mode):
                    # Update supply: remove one unit.
                    supply_data[supply_key]["quantity"] -= 1
                    total_dispatched += 1
                else:
                    logger.error("Dispatch error for {} at {} {}", etype, assigned_call["city"], assigned_call["county"])
    return total_dispatched

def run_simulation(params: SimulationParams) -> None:
    session = create_session(params.api_url)

    # Register signal handlers for graceful shutdown.
    def handle_shutdown(signum, frame):
        logger.info("Received shutdown signal: {}. Initiating graceful shutdown.", signum)
        STOP_EVENT.set()
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Reset simulation on backend.
    reset_url = f"{session.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = session.post(reset_url)
    if not r.ok:
        logger.error("Reset failed: {} {}", r.status_code, r.text)
        return
    try:
        logger.info("Simulation reset: {}", r.json())
    except Exception as e:
        logger.error("Error parsing reset response: {}", e)
        return

    # Load supply and build KDTree.
    supplies = initialize_supply(session, params.emergency_types)

    # Shared locks for updating active counters and supplies.
    active_lock = threading.Lock()
    supply_lock = threading.Lock()
    local_dispatch_count = 0
    last_status_check = time.time()

    # Polling backoff if no emergency is returned.
    consecutive_no_emergency = 0
    max_poll_sleep = 5  # seconds

    # Batch processing loop: fetch until maxActiveCalls are reached, then process via Hungarian algorithm.
    while not STOP_EVENT.is_set():
        PAUSE_EVENT.wait()
        batch = []
        # Fetch emergency calls until batch is full or no call returned.
        while len(batch) < params.maxActiveCalls and not STOP_EVENT.is_set():
            emergency = get_next_emergency(session)
            if emergency:
                batch.append(emergency)
                consecutive_no_emergency = 0  # Reset backoff counter.
            else:
                consecutive_no_emergency += 1
                sleep_time = min(params.poll_interval * (2 ** consecutive_no_emergency), max_poll_sleep)
                time.sleep(sleep_time)
        if not batch:
            # No emergencies available – check if remote target reached.
            if time.time() - last_status_check >= params.status_interval:
                status = get_status(session)
                if status and status.get("totalDispatches", 0) >= params.targetDispatches:
                    logger.info("Remote target reached.")
                    break
                last_status_check = time.time()
            continue

        # Process the batch via Hungarian assignment.
        dispatched = process_batch(session, batch, supplies, params, supply_lock)
        local_dispatch_count += dispatched
        logger.info("Processed batch of {} calls, dispatched {} units. Local total: {}.",
                    len(batch), dispatched, local_dispatch_count)

        # Check if target reached.
        if local_dispatch_count >= params.targetDispatches:
            logger.info("Local target reached: {}.", local_dispatch_count)
            break
        # Optionally check global status.
        if time.time() - last_status_check >= params.status_interval:
            status = get_status(session)
            if status and status.get("totalDispatches", 0) >= params.targetDispatches:
                logger.info("Remote target reached.")
                break
            last_status_check = time.time()

    # Stop simulation on backend.
    stop_resp = session.post(f"{session.base_url}/control/stop")
    if stop_resp.ok:
        try:
            logger.info("Simulation stopped: {}", stop_resp.json())
        except Exception as e:
            logger.error("Error parsing stop response: {}", e)
    else:
        logger.error("Stop failed: {} {}", stop_resp.status_code, stop_resp.text)
    STOP_EVENT.clear()

if __name__ == "__main__":
    params = SimulationParams(seed="mySeed", targetDispatches=100, maxActiveCalls=15)
    run_simulation(params)
