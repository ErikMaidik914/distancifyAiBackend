import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree

PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()

class SimulationParams:
    def __init__(self, api_url=None, seed="default", targetDispatches=100, maxActiveCalls=15, poll_interval=0.3, status_interval=5, emergency_types=None, debug_mode=False):
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.emergency_types = emergency_types or ["Medical", "Fire", "Police", "Rescue", "Utility"]
        self.debug_mode = debug_mode

def create_session(base_url, retries=3, backoff=0.5):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500,502,503,504])
    adapter = HTTPAdapter(max_retries=r)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.base_url = base_url
    return s

def request_with_retry(session, method, url, max_retries=3, backoff=0.5, **kwargs):
    for attempt in range(max_retries):
        try:
            response = session.request(method, url, **kwargs)
            if response.ok:
                return response
            if response.text and "Not started" in response.text:
                logger.error("Request {} {} returned 'Not started'.", method, url)
                return None
            logger.error("Request {} {} failed attempt {}: {} {}", method, url, attempt+1, response.status_code, response.text)
        except Exception as e:
            logger.error("Request {} {} error attempt {}: {}", method, url, attempt+1, e)
        time.sleep(backoff * (2 ** attempt))
    return None

def initialize_supply(session, emergency_types):
    supplies = {}
    for etype in emergency_types:
        url = f"{session.base_url}/{etype.lower()}/search"
        resp = request_with_retry(session, "GET", url, timeout=5)
        if resp:
            data = resp.json()
            local_supply = []
            supply_keys = []
            supply_points = []
            for entry in data:
                qty = entry.get("quantity", 0)
                lat = entry.get("latitude")
                lon = entry.get("longitude")
                county = entry.get("county")
                city = entry.get("city")
                if qty is None or qty < 0:
                    qty = 0
                if lat is None or lon is None or not county or not city or county.strip() == "":
                    continue
                key = (county.strip(), city.strip())
                local_supply.append({"key": key, "quantity": qty, "lat": lat, "lon": lon})
                supply_keys.append(key)
                supply_points.append((lat, lon))
            tree = KDTree(supply_points) if supply_points else None
            supplies[etype] = {"local_supply": local_supply, "supply_keys": supply_keys, "supply_points": supply_points, "tree": tree}
            logger.info("{} supply loaded.", etype)
        else:
            logger.error("{} supply loading failed.", etype)
    return supplies

def refresh_supply_by_city(session, etype, county, city):
    url = f"{session.base_url}/{etype.lower()}/searchbycity?county={county}&city={city}"
    resp = request_with_retry(session, "GET", url, timeout=3)
    if resp:
        data = resp.json()
        total = 0
        if isinstance(data, list):
            for entry in data:
                qty = entry.get("quantity", 0)
                if qty is None or qty < 0:
                    qty = 0
                total += qty
        elif isinstance(data, dict):
            qty = data.get("quantity", 0)
            if qty is None or qty < 0:
                qty = 0
            total = qty
        elif isinstance(data, int):
            total = data
        return total
    logger.error("Failed to refresh supply for {} {} {}.", etype, county, city)
    return 0

def dispatch(session, etype, srcCounty, srcCity, tgtCounty, tgtCity, qty, debug_mode=False):
    url = f"{session.base_url}/{etype.lower()}/dispatch"
    payload = {"sourceCounty": srcCounty, "sourceCity": srcCity, "targetCounty": tgtCounty, "targetCity": tgtCity, "quantity": qty}
    resp = request_with_retry(session, "POST", url, json=payload, timeout=5)
    if resp:
        if debug_mode:
            logger.debug("Dispatched {} {} from {} {} to {} {}.", qty, etype, srcCity, srcCounty, tgtCity, tgtCounty)
        return True
    logger.error("Dispatch failed for {} from {} {} to {} {}.", etype, srcCity, srcCounty, tgtCity, tgtCounty)
    return False

def get_next_emergency(session):
    url = f"{session.base_url}/calls/next"
    resp = request_with_retry(session, "GET", url, timeout=5)
    if resp:
        if resp.status_code == 404:
            return None
        return resp.json()
    logger.error("Error calling /calls/next.")
    return None

def get_status(session):
    url = f"{session.base_url}/control/status"
    resp = request_with_retry(session, "GET", url, timeout=5)
    if resp:
        return resp.json()
    logger.error("Error fetching status.")
    return None

def run_simulation(params):
    session = create_session(params.api_url)
    reset_url = f"{session.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = request_with_retry(session, "POST", reset_url, timeout=5)
    if r is None:
        logger.error("Reset failed.")
        return
    logger.info("Simulation reset: {}", r.json())
    supplies = initialize_supply(session, params.emergency_types)
    active_lock = threading.Lock()
    supply_lock = threading.Lock()
    active_count = 0
    local_dispatch_count = 0
    pool = ThreadPoolExecutor(max_workers=40)
    futures = []
    last_status_check = time.time()
    stop_fetching = False
    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        for req in call.get("requests", []):
            needed = req.get("Quantity", 0)
            if needed <= 0:
                logger.debug("No {} units required at {} {}.", req.get("Type"), call.get("city"), call.get("county"))
                continue
            supply_data = supplies.get(req.get("Type"))
            if not supply_data or not supply_data["tree"]:
                logger.error("No supply available for {}.", req.get("Type"))
                continue
            pt = (call.get("latitude"), call.get("longitude"))
            distances, indices = supply_data["tree"].query(pt, k=len(supply_data["supply_points"]))
            remaining = needed
            with supply_lock:
                for idx in indices:
                    key = supply_data["supply_keys"][idx]
                    current_qty = refresh_supply_by_city(session, req.get("Type"), key[0], key[1])
                    for supply in supply_data["local_supply"]:
                        if supply["key"] == key:
                            supply["quantity"] = current_qty
                            break
                    if current_qty <= 0:
                        continue
                    use = min(current_qty, remaining)
                    if dispatch(session, req.get("Type"), key[0], key[1], call.get("county"), call.get("city"), use, params.debug_mode):
                        for supply in supply_data["local_supply"]:
                            if supply["key"] == key:
                                supply["quantity"] -= use
                                break
                        remaining -= use
                        local_dispatch_count += use
                        if params.debug_mode:
                            logger.debug("Dispatch count updated: {}", local_dispatch_count)
                        if remaining <= 0:
                            break
                    else:
                        logger.error("Dispatch error for {} at {} {}.", req.get("Type"), call.get("city"), call.get("county"))
            if remaining > 0:
                logger.warning("Not fully dispatched for {} at {} {}; missing {} units.", req.get("Type"), call.get("city"), call.get("county"), remaining)
        with active_lock:
            active_count -= 1
            logger.info("Processed emergency at {} {}. Active: {}", call.get("city"), call.get("county"), active_count)
    while True:
        if STOP_EVENT.is_set():
            break
        PAUSE_EVENT.wait()
        if local_dispatch_count >= params.targetDispatches:
            stop_fetching = True
            pending = get_next_emergency(session)
            if pending:
                logger.warning("Pending emergency in queue: {}. Waiting for it to be processed.", pending)
                time.sleep(params.poll_interval)
                continue
            else:
                logger.info("Target reached and no pending emergencies.")
                break
        if not stop_fetching:
            with active_lock:
                current = active_count
            if current < params.maxActiveCalls:
                emergency = get_next_emergency(session)
                if emergency:
                    with active_lock:
                        active_count += 1
                    futures.append(pool.submit(process_emergency, emergency))
                    logger.info("Submitted emergency at {} {}. Active: {}", emergency.get("city"), emergency.get("county"), active_count)
                else:
                    time.sleep(params.poll_interval)
            else:
                time.sleep(params.poll_interval)
        futures = [f for f in futures if not f.done()]
        if time.time() - last_status_check >= params.status_interval:
            status = get_status(session)
            if status and status.get("totalDispatches", 0) >= params.targetDispatches:
                logger.info("Remote target reached.")
                stop_fetching = True
            last_status_check = time.time()
    stop_resp = request_with_retry(session, "POST", f"{session.base_url}/control/stop", timeout=5)
    if stop_resp:
        logger.info("Simulation stopped: {}", stop_resp.json())
    else:
        logger.error("Stop failed.")
    STOP_EVENT.clear()

if __name__ == "__main__":
    params = SimulationParams(seed="mySeed", targetDispatches=10, maxActiveCalls=2)
    run_simulation(params)
