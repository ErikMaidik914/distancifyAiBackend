import time
import threading
import requests
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from scipy.spatial import KDTree

BASE = "http://localhost:5000"
DEBUG_MODE = False

# Global shared state and locks for thread safety
session = requests.Session()
active_lock = threading.Lock()
supply_lock = threading.Lock()

active_count = 0
local_dispatch_count = 0
local_supply = {}
supply_points = []
supply_keys = []
kdtree = None

class SimulationParams:
    """
    Class to encapsulate simulation parameters.
    """
    def __init__(self, seed="default", targetDispatches=100, maxActiveCalls=15, poll_interval=0.3, status_interval=5):
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval

def initialize_supply():
    """
    Initializes local supply data and builds a KDTree for spatial queries.
    """
    global kdtree
    r = session.get(f"{BASE}/medical/search")
    if r.ok:
        data = r.json()
        with supply_lock:
            for entry in data:
                key = (entry["county"], entry["city"])
                local_supply[key] = {
                    "quantity": entry["quantity"],
                    "latitude": entry["latitude"],
                    "longitude": entry["longitude"]
                }
                supply_keys.append(key)
                supply_points.append((entry["latitude"], entry["longitude"]))
        kdtree = KDTree(supply_points)
        logger.info("Supply loaded and KDTree built.")
    else:
        logger.error("Supply loading failed: {} {}", r.status_code, r.text)
        raise Exception("Supply loading failed")

def dispatch(srcCounty, srcCity, tgtCounty, tgtCity, qty):
    """
    Dispatches units from a source to a target location.
    """
    data = {
        "sourceCounty": srcCounty,
        "sourceCity": srcCity,
        "targetCounty": tgtCounty,
        "targetCity": tgtCity,
        "quantity": qty
    }
    r = session.post(f"{BASE}/medical/dispatch", json=data)
    if r.ok:
        if DEBUG_MODE:
            logger.debug("Dispatched {} from {} {} to {} {}", qty, srcCity, srcCounty, tgtCity, tgtCounty)
        return True
    logger.error("Dispatch failed from {} {} to {} {}: {} {}", srcCity, srcCounty, tgtCity, tgtCounty, r.status_code, r.text)
    return False

def process_emergency(call, broadcast):
    """
    Processes an emergency call by determining the required units, dispatching them and broadcasting progress updates.
    
    Args:
        call (dict): Emergency call data.
        broadcast (function): Callable to broadcast JSON updates asynchronously.
    """
    global local_dispatch_count, active_count
    needed = sum(req["Quantity"] for req in call.get("requests", []))
    if needed <= 0:
        with active_lock:
            active_count -= 1
        return

    pt = (call["latitude"], call["longitude"])
    distances, indices = kdtree.query(pt, k=len(supply_points))
    remaining = needed

    with supply_lock:
        for idx in indices:
            key = supply_keys[idx]
            available = local_supply[key]["quantity"]
            if available <= 0:
                continue
            use = min(available, remaining)
            if dispatch(key[0], key[1], call["county"], call["city"], use):
                local_supply[key]["quantity"] -= use
                remaining -= use
                local_dispatch_count += use
                if remaining <= 0:
                    break
            else:
                logger.error("Dispatch error at {} {}.", call["city"], call["county"])

    if remaining > 0:
        logger.warning("Not fully dispatched at {} {}; missing {} units.", call["city"], call["county"], remaining)

    with active_lock:
        active_count -= 1

    # Broadcast a progress update asynchronously
    update = {
        "event": "update",
        "active_count": active_count,
        "local_dispatch_count": local_dispatch_count,
        "city": call["city"],
        "county": call["county"],
        "timestamp": time.time()
    }
    asyncio.run_coroutine_threadsafe(broadcast(json.dumps(update)), asyncio.get_event_loop())

def get_next_emergency():
    """
    Retrieves the next emergency call from the external service.
    """
    r = session.get(f"{BASE}/calls/next")
    if r.status_code == 404:
        return None
    if r.ok:
        try:
            return r.json()
        except Exception as e:
            logger.error("JSON parse error: {} {}", e, r.text)
    else:
        logger.error("Error calling /calls/next: {} {}", r.status_code, r.text)
    return None

def get_status():
    """
    Retrieves the current status from the external service.
    """
    r = session.get(f"{BASE}/control/status")
    if r.ok:
        return r.json()
    logger.error("Error fetching status: {} {}", r.status_code, r.text)
    return None

def run_simulation(params: SimulationParams, broadcast):
    """
    Runs the simulation task in a separate thread.
    
    Args:
        params (SimulationParams): The parameters for simulation.
        broadcast (function): Callable to broadcast simulation updates over websockets.
    """
    global active_count, local_dispatch_count, local_supply, supply_points, supply_keys, kdtree
    local_dispatch_count = 0
    active_count = 0
    local_supply = {}
    supply_points = []
    supply_keys = []
    kdtree = None

    reset_url = f"{BASE}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = session.post(reset_url)
    if not r.ok:
        logger.error("Reset failed: {} {}", r.status_code, r.text)
        return
    logger.info("Simulation reset: {}", r.json())

    try:
        initialize_supply()
    except Exception as e:
        logger.error("Initialization error: {}", e)
        return

    executor = ThreadPoolExecutor(max_workers=params.maxActiveCalls)
    futures = []
    last_status_check = time.time()

    while True:
        if local_dispatch_count >= params.targetDispatches:
            logger.info("Local target reached: {}.", local_dispatch_count)
            break

        if time.time() - last_status_check >= params.status_interval:
            status = get_status()
            if status and status.get("totalDispatches", 0) >= params.targetDispatches:
                logger.info("Remote target reached.")
                break
            last_status_check = time.time()

        with active_lock:
            current = active_count

        if current < params.maxActiveCalls:
            emergency = get_next_emergency()
            if emergency:
                with active_lock:
                    active_count += 1
                futures.append(executor.submit(process_emergency, emergency, broadcast))
                update = {
                    "event": "update",
                    "active_count": active_count,
                    "local_dispatch_count": local_dispatch_count,
                    "city": emergency["city"],
                    "county": emergency["county"],
                    "timestamp": time.time()
                }
                asyncio.run_coroutine_threadsafe(broadcast(json.dumps(update)), asyncio.get_event_loop())
            else:
                time.sleep(params.poll_interval)
        else:
            time.sleep(params.poll_interval)
        futures = [f for f in futures if not f.done()]

    stop = session.post(f"{BASE}/control/stop")
    if stop.ok:
        logger.info("Simulation stopped: {}", stop.json())
    else:
        logger.error("Stop failed: {} {}", stop.status_code, stop.text)

    # Broadcast simulation completion
    completion = {
        "event": "complete",
        "local_dispatch_count": local_dispatch_count,
        "timestamp": time.time()
    }
    asyncio.run_coroutine_threadsafe(broadcast(json.dumps(completion)), asyncio.get_event_loop())
