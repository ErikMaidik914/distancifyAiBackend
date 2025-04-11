import requests, math, time, threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from loguru import logger
from scipy.spatial import KDTree

BASE = "http://localhost:5000"
DEBUG_MODE = False

session = requests.Session()
active_lock = threading.Lock()
supply_lock = threading.Lock()
active_count = 0
local_dispatch_count = 0
local_supply = {}
supply_points = []
supply_keys = []
kdtree = None

def initialize_supply():
    global kdtree
    r = session.get(f"{BASE}/medical/search")
    if r.ok:
        data = r.json()
        with supply_lock:
            for entry in data:
                key = (entry["county"], entry["city"])
                local_supply[key] = {"quantity": entry["quantity"], "latitude": entry["latitude"], "longitude": entry["longitude"]}
                supply_keys.append(key)
                supply_points.append((entry["latitude"], entry["longitude"]))
        kdtree = KDTree(supply_points)
        logger.info("Supply loaded and KDTree built.")
    else:
        logger.error("Supply loading failed: {} {}", r.status_code, r.text)

def dispatch(srcCounty, srcCity, tgtCounty, tgtCity, qty):
    data = {"sourceCounty": srcCounty, "sourceCity": srcCity, "targetCounty": tgtCounty, "targetCity": tgtCity, "quantity": qty}
    r = session.post(f"{BASE}/medical/dispatch", json=data)
    if r.ok:
        if DEBUG_MODE:
            logger.debug("Dispatched {} from {} {} to {} {}", qty, srcCity, srcCounty, tgtCity, tgtCounty)
        return True
    logger.error("Dispatch failed from {} {} to {} {}: {} {}", srcCity, srcCounty, tgtCity, tgtCounty, r.status_code, r.text)
    return False

def process_emergency(call):
    global local_dispatch_count, active_count
    needed = sum(req["Quantity"] for req in call.get("requests", []))
    if needed <= 0:
        logger.debug("No ambulances required at {} {}.", call["city"], call["county"])
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
                if DEBUG_MODE:
                    logger.debug("Dispatch count updated: {}", local_dispatch_count)
                if remaining <= 0:
                    break
            else:
                logger.error("Dispatch error at {} {}.", call["city"], call["county"])
    if remaining > 0:
        logger.warning("Not fully dispatched at {} {}; missing {} ambulances.", call["city"], call["county"], remaining)
    with active_lock:
        active_count -= 1
        logger.info("Processed emergency at {} {}. Active: {}", call["city"], call["county"], active_count)

def get_next_emergency():
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
        if local_dispatch_count >= targetDispatches:
            logger.info("Local target reached: {}.", local_dispatch_count)
            break
        if time.time() - last_status_check >= status_interval:
            status = get_status()
            if status and status.get("totalDispatches", 0) >= targetDispatches:
                logger.info("Remote target reached.")
                break
            last_status_check = time.time()
        with active_lock:
            current = active_count
        if current < maxActiveCalls:
            emergency = get_next_emergency()
            if emergency:
                with active_lock:
                    active_count += 1
                futures.append(executor.submit(process_emergency, emergency))
                logger.info("Submitted emergency at {} {}. Active: {}", emergency["city"], emergency["county"], active_count)
            else:
                time.sleep(poll_interval)
        else:
            time.sleep(poll_interval)
        futures = [f for f in futures if not f.done()]
    stop = session.post(f"{BASE}/control/stop")
    if stop.ok:
        logger.info("Simulation stopped: {}", stop.json())
    else:
        logger.error("Stop failed: {} {}", stop.status_code, stop.text)

if __name__ == "__main__":
    main("mySeed", 10000, 1000)
