# simulation.py
import os, time, threading, json, asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree

class SimulationParams:
    def __init__(
        self,
        api_url=None,
        seed="default",
        targetDispatches=100,
        maxActiveCalls=15,
        poll_interval=0.3,
        status_interval=5,
        debug=False
    ):
        self.api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.debug = debug

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

class Simulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.session = create_session(params.api_url)
        self.emergency_types = ["Medical", "Fire", "Police", "Rescue", "Utility"]
        self.supplies = {}
        self.active_lock = threading.Lock()
        self.supply_lock = threading.Lock()
        self.active_count = 0
        self.local_dispatch_count = 0
        self.executor = ThreadPoolExecutor(max_workers=self.params.maxActiveCalls)
        self.running = False

    def initialize_supply(self):
        for etype in self.emergency_types:
            url = f"{self.session.base_url}/{etype.lower()}/search"
            resp = self.session.get(url)
            if resp.ok:
                data = resp.json()
                local_supply = {}
                supply_keys = []
                supply_points = []
                for entry in data:
                    key = (entry["county"], entry["city"])
                    local_supply[key] = {
                        "quantity": entry["quantity"],
                        "latitude": entry["latitude"],
                        "longitude": entry["longitude"],
                    }
                    supply_keys.append(key)
                    supply_points.append((entry["latitude"], entry["longitude"]))
                kdtree = KDTree(supply_points) if supply_points else None
                self.supplies[etype] = {
                    "local_supply": local_supply,
                    "supply_keys": supply_keys,
                    "supply_points": supply_points,
                    "kdtree": kdtree
                }
                logger.info(f"{etype} supply loaded and KDTree built.")
            else:
                logger.error(f"{etype} supply loading failed: {resp.status_code} {resp.text}")

    def dispatch(self, etype, srcCounty, srcCity, tgtCounty, tgtCity, qty):
        url = f"{self.session.base_url}/{etype.lower()}/dispatch"
        data = {
            "sourceCounty": srcCounty,
            "sourceCity": srcCity,
            "targetCounty": tgtCounty,
            "targetCity": tgtCity,
            "quantity": qty
        }
        resp = self.session.post(url, json=data)
        if resp.ok:
            if self.params.debug:
                logger.debug(f"Dispatched {qty} {etype} from {srcCity} {srcCounty} to {tgtCity} {tgtCounty}")
            return True
        logger.error(f"Dispatch failed for {etype} from {srcCity} {srcCounty} to {tgtCity} {tgtCounty}: {resp.status_code} {resp.text}")
        return False

    def process_emergency(self, call, broadcast):
        for req in call.get("requests", []):
            needed = req["Quantity"]
            if needed <= 0:
                continue
            supply_data = self.supplies.get(req["Type"])
            if not supply_data or not supply_data["kdtree"]:
                logger.error(f"No supply available for {req['Type']}.")
                continue

            pt = (call["latitude"], call["longitude"])
            dist, indices = supply_data["kdtree"].query(pt, k=len(supply_data["supply_points"]))
            remaining = needed

            with self.supply_lock:
                for idx in indices:
                    key = supply_data["supply_keys"][idx]
                    available = supply_data["local_supply"][key]["quantity"]
                    if available <= 0:
                        continue
                    use = min(available, remaining)
                    ok = self.dispatch(req["Type"], key[0], key[1], call["county"], call["city"], use)
                    if ok:
                        supply_data["local_supply"][key]["quantity"] -= use
                        remaining -= use
                        self.local_dispatch_count += use
                        if remaining <= 0:
                            break

        with self.active_lock:
            self.active_count -= 1

        msg = {
            "event": "update",
            "city": call["city"],
            "county": call["county"],
            "active_count": self.active_count,
            "local_dispatch_count": self.local_dispatch_count,
            "timestamp": time.time()
        }
        asyncio.run_coroutine_threadsafe(broadcast(json.dumps(msg)), asyncio.get_event_loop())

    def get_next_emergency(self):
        url = f"{self.session.base_url}/calls/next"
        resp = self.session.get(url)
        if resp.status_code == 404:
            return None
        if resp.ok:
            return resp.json()
        logger.error(f"Error calling /calls/next: {resp.status_code} {resp.text}")
        return None

    def get_remote_status(self):
        url = f"{self.session.base_url}/control/status"
        resp = self.session.get(url)
        if resp.ok:
            return resp.json()
        logger.error(f"Error fetching status: {resp.status_code} {resp.text}")
        return None

    def run_simulation(self, broadcast):
        if self.running:
            logger.error("Simulation is already running.")
            return
        self.running = True
        self.active_count = 0
        self.local_dispatch_count = 0
        self.supplies.clear()

        reset_url = (
            f"{self.session.base_url}/control/reset?seed={self.params.seed}"
            f"&targetDispatches={self.params.targetDispatches}"
            f"&maxActiveCalls={self.params.maxActiveCalls}"
        )
        r = self.session.post(reset_url)
        if not r.ok:
            logger.error(f"Reset failed: {r.status_code} {r.text}")
            self.running = False
            return
        logger.info(f"Simulation reset: {r.json()}")

        self.initialize_supply()
        futures = []
        last_status_check = time.time()

        while True:
            if self.local_dispatch_count >= self.params.targetDispatches:
                logger.info(f"Local target reached: {self.local_dispatch_count}.")
                break

            if time.time() - last_status_check >= self.params.status_interval:
                st = self.get_remote_status()
                if st and st.get("totalDispatches", 0) >= self.params.targetDispatches:
                    logger.info("Remote target reached.")
                    break
                last_status_check = time.time()

            with self.active_lock:
                current = self.active_count
            if current < self.params.maxActiveCalls:
                call = self.get_next_emergency()
                if call:
                    with self.active_lock:
                        self.active_count += 1
                    futures.append(self.executor.submit(self.process_emergency, call, broadcast))
                    msg = {
                        "event": "update",
                        "city": call["city"],
                        "county": call["county"],
                        "active_count": self.active_count,
                        "local_dispatch_count": self.local_dispatch_count,
                        "timestamp": time.time()
                    }
                    asyncio.run_coroutine_threadsafe(broadcast(json.dumps(msg)), asyncio.get_event_loop())
                else:
                    time.sleep(self.params.poll_interval)
            else:
                time.sleep(self.params.poll_interval)

            futures = [f for f in futures if not f.done()]

        stop_resp = self.session.post(f"{self.session.base_url}/control/stop")
        if stop_resp.ok:
            logger.info(f"Simulation stopped: {stop_resp.json()}")
        else:
            logger.error(f"Stop failed: {stop_resp.status_code} {stop_resp.text}")

        msg = {
            "event": "complete",
            "local_dispatch_count": self.local_dispatch_count,
            "timestamp": time.time()
        }
        asyncio.run_coroutine_threadsafe(broadcast(json.dumps(msg)), asyncio.get_event_loop())
        self.running = False

    def get_local_status(self):
        return {
            "active_count": self.active_count,
            "local_dispatch_count": self.local_dispatch_count
        }