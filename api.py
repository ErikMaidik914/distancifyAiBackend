import requests
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from loguru import logger
from scipy.spatial import KDTree

# Configuration
BASE = "http://localhost:5000"
DEBUG_MODE = False  # Set to True for detailed logging

session = requests.Session()

# Global counters and locks
active_lock = threading.Lock()
active_count = 0
local_dispatch_count = 0  # Local counter for successful dispatches

# Local supply data (cached once)
local_supply = {}  # Key: (county, city) -> { 'quantity', 'latitude', 'longitude' }
supply_lock = threading.Lock()
supply_points = []  # List of (latitude, longitude)
supply_keys = []    # List of (county, city)
kdtree = None

def dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def initialize_supply():
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
        global kdtree
        kdtree = KDTree(supply_points)
        logger.info("Initial supply loaded and KDTree built; using local cache.")
    else:
        logger.error("Initial supply loading failed: {} {}", r.status_code, r.text)

def dispatch(srcCounty, srcCity, tgtCounty, tgtCity, qty):
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
    else:
        logger.error("Dispatch failed from {} {} to {} {}: {} {}", srcCity, srcCounty, tgtCity, tgtCounty, r.status_code, r.text)
        return False

def process_emergency(call):
    global local_dispatch_count
    needed = sum(req["Quantity"] for req in call.get("requests", []))
    if needed <= 0:
        logger.debug("Emergency at {} {} requires no ambulances.", call["city"], call["county"])
        return

    emergency_point = (call["latitude"], call["longitude"])
    distances, indices = kdtree.query(emergency_point, k=len(supply_points))
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
                if DEBUG_MODE:
                    logger.debug("Local dispatch count updated: {}", local_dispatch_count)
                if remaining <= 0:
                    break
            else:
                logger.error("Dispatch error for emergency at {} {}.", call["city"], call["county"])
    if remaining > 0:
        logger.warning("Emergency at {} {} not fully dispatched; missing {} ambulances.", call["city"], call["county"], remaining)

    with active_lock:
        global active_count
        active_count -= 1
        logger.info("Processed emergency at {} {}. Active emergencies: {}", call["city"], call["county"], active_count)

def get_next_emergency():
    r = session.get(f"{BASE}/calls/next")
    if r.status_code == 404:
        return None
    elif r.ok:
        try:
            return r.json()
        except Exception as e:
            logger.error("JSON parse error from /calls/next: {} {}", e, r.text)
    else:
        logger.error("Error calling /calls/next: {} {}", r.status_code, r.text)
    return None

def get_status():
    r = session.get(f"{BASE}/control/status")
    if r.ok:
        return r.json()
    logger.error("Error fetching status: {} {}", r.status_code, r.text)
    return None

def main(seed="default", targetDispatches=100, maxActiveCalls=15, poll_interval=0.3, status_interval=5):
    reset_url = f"{BASE}/control/reset?seed={seed}&targetDispatches={targetDispatches}&maxActiveCalls={maxActiveCalls}"
    r = session.post(reset_url)
    if not r.ok:
        logger.error("Reset failed: {} {}", r.status_code, r.text)
        return
    logger.info("Simulation reset: {}", r.json())

    initialize_supply()
    executor = ThreadPoolExecutor(max_workers=maxActiveCalls)
    futures = []
    global active_count, local_dispatch_count

    last_status_check = time.time()
    while True:
        # Local check using our own counter.
        if local_dispatch_count >= targetDispatches:
            logger.info("Local target dispatches reached: {}. Stopping simulation.", local_dispatch_count)
            break

        # Perform a status check less frequently for cross-validation.
        if time.time() - last_status_check >= status_interval:
            status = get_status()
            if status:
                remote_dispatches = status.get("totalDispatches", 0)
                logger.info("Status check: remote dispatches = {} (local = {})", remote_dispatches, local_dispatch_count)
                if remote_dispatches >= targetDispatches:
                    logger.info("Remote target dispatches reached. Stopping simulation.")
                    break
            last_status_check = time.time()

        with active_lock:
            current_active = active_count
        if current_active < maxActiveCalls:
            emergency = get_next_emergency()
            if emergency:
                with active_lock:
                    active_count += 1
                futures.append(executor.submit(process_emergency, emergency))
                logger.info("Submitted emergency at {} {}. Active count: {}",
                            emergency["city"], emergency["county"], active_count)
            else:
                time.sleep(poll_interval)
        else:
            time.sleep(poll_interval)

        if futures:
            done, not_done = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            futures = list(not_done)

    stop = session.post(f"{BASE}/control/stop")
    if stop.ok:
        logger.info("Simulation stopped: {}", stop.json())
    else:
        logger.error("Stop failed: {} {}", stop.status_code, stop.text)

if __name__ == "__main__":
    main("mySeed", 10000, 1000)
