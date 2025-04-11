import requests, math, time, threading
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from scipy.spatial import KDTree

BASE = "http://localhost:5000"
DEBUG_MODE = False

session = requests.Session()
active_lock = threading.Lock()
supply_lock = threading.Lock()
active_count = 0
local_dispatch_count = 0
emergency_types = ["Medical", "Fire", "Police", "Rescue", "Utility"]
supplies = {}

def initialize_supply():
    global supplies
    for etype in emergency_types:
        endpoint = f"{BASE}/{etype.lower()}/search"
        r = session.get(endpoint)
        if r.ok:
            data = r.json()
            local_supply = {}
            supply_keys = []
            supply_points = []
            for entry in data:
                key = (entry["county"], entry["city"])
                local_supply[key] = {
                    "quantity": entry["quantity"],
                    "latitude": entry["latitude"],
                    "longitude": entry["longitude"]
                }
                supply_keys.append(key)
                supply_points.append((entry["latitude"], entry["longitude"]))
            kdtree = KDTree(supply_points) if supply_points else None
            supplies[etype] = {
                "local_supply": local_supply,
                "supply_keys": supply_keys,
                "supply_points": supply_points,
                "kdtree": kdtree
            }
            logger.info("{} supply loaded and KDTree built.", etype)
        else:
            logger.error("{} supply loading failed: {} {}", etype, r.status_code, r.text)

def dispatch(etype, srcCounty, srcCity, tgtCounty, tgtCity, qty):
    data = {
        "sourceCounty": srcCounty,
        "sourceCity": srcCity,
        "targetCounty": tgtCounty,
        "targetCity": tgtCity,
        "quantity": qty
    }
    endpoint = f"{BASE}/{etype.lower()}/dispatch"
    r = session.post(endpoint, json=data)
    if r.ok:
        if DEBUG_MODE:
            logger.debug("Dispatched {} {} from {} {} to {} {}", qty, etype, srcCity, srcCounty, tgtCity, tgtCounty)
        return True
    logger.error("Dispatch failed for {} from {} {} to {} {}: {} {}", etype, srcCity, srcCounty, tgtCity, tgtCounty, r.status_code, r.text)
    return False

def process_emergency(call):
    global local_dispatch_count, active_count
    for req in call.get("requests", []):
        needed = req["Quantity"]
        if needed <= 0:
            logger.debug("No {} units required at {} {}.", req["Type"], call["city"], call["county"])
            continue
        supply_data = supplies.get(req["Type"])
        if not supply_data or not supply_data["kdtree"]:
            logger.error("No supply available for {}.", req["Type"])
            continue
        pt = (call["latitude"], call["longitude"])
        distances, indices = supply_data["kdtree"].query(pt, k=len(supply_data["supply_points"]))
        remaining = needed
        with supply_lock:
            for idx in indices:
                key = supply_data["supply_keys"][idx]
                available = supply_data["local_supply"][key]["quantity"]
                if available <= 0:
                    continue
                use = min(available, remaining)
                if dispatch(req["Type"], key[0], key[1], call["county"], call["city"], use):
                    supply_data["local_supply"][key]["quantity"] -= use
                    remaining -= use
                    local_dispatch_count += use
                    if DEBUG_MODE:
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
