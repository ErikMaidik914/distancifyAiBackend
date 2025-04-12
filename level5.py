#!/usr/bin/env python3
import os
import time
import threading
import requests
import signal
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
from typing import Any, Dict, Optional
import argparse

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

class Simulation:
    def __init__(self, params: SimulationParams) -> None:
        self.params = params
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_event = threading.Event()
        self.active_lock = threading.Lock()
        self.supply_lock = threading.Lock()
        self.active_count: int = 0
        self.local_dispatch_count: int = 0
        self.supplies: Dict[str, Any] = {}
        self.last_status_check = time.time()
        self.session = self.create_session(self.params.api_url)
        self.configure_signal_handlers()
        self.consecutive_no_emergency = 0
        self.max_poll_sleep = 5

    def configure_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received: {}. Stopping simulation.", signum)
        self.stop_event.set()

    def create_session(self, base_url: str, retries: int = 3, backoff: float = 0.5) -> requests.Session:
        s = requests.Session()
        retry_strategy = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.base_url = base_url  # type: ignore
        s.headers.update({"Content-Type": "application/json"})
        return s

    def request_with_retry(self, method: str, url: str, max_retries: int = 3, backoff: float = 0.5, **kwargs) -> Optional[requests.Response]:
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                if "/calls/next" in url and response.status_code == 404:
                    return response
                if response.ok:
                    return response
                if response.status_code == 401:
                    logger.info("Token expired, refreshing token...")
                    if not self.auth_refresh():
                        logger.error("Token refresh failed.")
                        return None
                    continue
                logger.error("Request {} {} failed attempt {}: {} {}", method, url, attempt + 1, response.status_code, response.text)
            except Exception as e:
                logger.exception("Request {} {} error attempt {}: {}", method, url, attempt + 1, e)
            time.sleep(backoff * (2 ** attempt))
        return None

    def auth_login(self) -> Optional[Dict[str, Any]]:
        login_url = f"{self.session.base_url}/auth/login"
        payload = {"username": "distancify", "password": "hackathon"}
        response = self.request_with_retry("POST", login_url, json=payload, timeout=5)
        if response:
            try:
                data = response.json()
            except Exception as e:
                logger.exception("Error parsing login response: {}", e)
                return None
            self.session._token = data.get("token")
            self.session._refresh_token = data.get("refreshToken")
            if self.session._token:
                self.session.headers.update({"Authorization": "Bearer " + self.session._token})
                return data
        logger.error("Login failed: {}", response.text if response else "No response")
        return None

    def auth_refresh(self) -> Optional[Dict[str, Any]]:
        refresh_url = f"{self.session.base_url}/auth/refreshtoken"
        headers = {"refresh_token": self.session._refresh_token}
        response = self.request_with_retry("POST", refresh_url, headers=headers, timeout=5)
        if response:
            try:
                data = response.json()
            except Exception as e:
                logger.exception("Error parsing refresh response: {}", e)
                return None
            self.session._token = data.get("token")
            self.session._refresh_token = data.get("refreshToken")
            if self.session._token:
                self.session.headers.update({"Authorization": "Bearer " + self.session._token})
                return data
        logger.error("Refresh token failed: {}", response.text if response else "No response")
        return None

    def initialize_supply(self) -> None:
        supplies: Dict[str, Any] = {}
        for etype in self.params.emergency_types:
            url = f"{self.session.base_url}/{etype.lower()}/search"
            resp = self.request_with_retry("GET", url, timeout=5)
            if resp:
                try:
                    data = resp.json()
                except Exception as e:
                    logger.exception("Failed to parse JSON for {} supply: {}", etype, e)
                    continue
                local_supply, supply_keys, supply_points = [], [], []
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
        self.supplies = supplies

    def refresh_supply_by_city(self, etype: str, county: str, city: str) -> int:
        url = f"{self.session.base_url}/{etype.lower()}/searchbycity?county={county}&city={city}"
        resp = self.request_with_retry("GET", url, timeout=3)
        if resp:
            try:
                data = resp.json()
            except Exception as e:
                logger.exception("Error parsing JSON in refresh_supply_by_city: {}", e)
                return 0
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

    def dispatch(self, etype: str, srcCounty: str, srcCity: str, tgtCounty: str, tgtCity: str, qty: int) -> bool:
        url = f"{self.session.base_url}/{etype.lower()}/dispatch"
        payload = {"sourceCounty": srcCounty, "sourceCity": srcCity, "targetCounty": tgtCounty, "targetCity": tgtCity, "quantity": qty}
        resp = self.request_with_retry("POST", url, json=payload, timeout=5)
        if resp:
            if self.params.debug_mode:
                logger.debug("Dispatched {} {} from {} {} to {} {}.", qty, etype, srcCity, srcCounty, tgtCity, tgtCounty)
            return True
        logger.error("Dispatch failed for {} from {} {} to {} {}.", etype, srcCity, srcCounty, tgtCity, tgtCounty)
        return False

    def get_next_emergency(self) -> Optional[Dict[str, Any]]:
        url = f"{self.session.base_url}/calls/next"
        resp = self.request_with_retry("GET", url, timeout=5)
        if resp:
            if resp.status_code == 404:
                return None
            try:
                return resp.json()
            except Exception as e:
                logger.exception("Error parsing JSON in get_next_emergency: {}", e)
                return None
        logger.error("Error calling /calls/next.")
        return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        url = f"{self.session.base_url}/control/status"
        resp = self.request_with_retry("GET", url, timeout=5)
        if resp:
            try:
                return resp.json()
            except Exception as e:
                logger.exception("Error parsing JSON in get_status: {}", e)
                return None
        logger.error("Error fetching status.")
        return None

    def stop_simulation(self) -> None:
        stop_url = f"{self.session.base_url}/control/stop"
        stop_resp = self.request_with_retry("POST", stop_url, timeout=5)
        if stop_resp:
            try:
                logger.info("Simulation stopped: {}", stop_resp.json())
            except Exception as e:
                logger.exception("Error parsing stop response: {}", e)
        else:
            logger.error("Stop failed.")
        logger.remove()
        self.stop_event.clear()

    def process_emergency(self, call: Dict[str, Any]) -> None:
        try:
            for req in call.get("requests", []):
                needed = req.get("Quantity", 0)
                if needed <= 0:
                    logger.debug("No {} units required at {} {}.", req.get("Type"), call.get("city"), call.get("county"))
                    continue
                supply_data = self.supplies.get(req.get("Type"))
                if not supply_data or not supply_data.get("tree"):
                    logger.error("No supply available for {}.", req.get("Type"))
                    continue
                pt = (call.get("latitude"), call.get("longitude"))
                distances, indices = supply_data["tree"].query(pt, k=len(supply_data["supply_points"]))
                remaining = needed
                with self.supply_lock:
                    for idx in indices:
                        key = supply_data["supply_keys"][idx]
                        current_qty = self.refresh_supply_by_city(req.get("Type"), key[0], key[1])
                        for supply in supply_data["local_supply"]:
                            if supply["key"] == key:
                                supply["quantity"] = current_qty
                                break
                        if current_qty <= 0:
                            continue
                        use = min(current_qty, remaining)
                        if self.dispatch(req.get("Type"), key[0], key[1], call.get("county"), call.get("city"), use):
                            for supply in supply_data["local_supply"]:
                                if supply["key"] == key:
                                    supply["quantity"] -= use
                                    break
                            remaining -= use
                            self.local_dispatch_count += use
                            if self.params.debug_mode:
                                logger.debug("Dispatch count updated: {}", self.local_dispatch_count)
                            if remaining <= 0:
                                break
                        else:
                            logger.error("Dispatch error for {} at {} {}.", req.get("Type"), call.get("city"), call.get("county"))
                if remaining > 0:
                    logger.warning("Not fully dispatched for {} at {} {}; missing {} units.", req.get("Type"), call.get("city"), call.get("county"), remaining)
        except Exception as e:
            logger.exception("Error processing emergency {}: {}", call, e)
        finally:
            with self.active_lock:
                self.active_count -= 1
                logger.info("Processed emergency at {} {}. Active: {}", call.get("city"), call.get("county"), self.active_count)

    def run(self) -> None:
        try:
            reset_url = f"{self.session.base_url}/control/reset?seed={self.params.seed}&targetDispatches={self.params.targetDispatches}&maxActiveCalls={self.params.maxActiveCalls}"
            r = self.request_with_retry("POST", reset_url, timeout=5)
            if not r:
                logger.error("Reset failed.")
                return
            logger.info("Simulation reset: {}", r.json())
            if not self.auth_login():
                logger.error("Login failed.")
                return
            self.initialize_supply()
            with ThreadPoolExecutor(max_workers=1000) as pool:
                while True:
                    if self.stop_event.is_set():
                        break
                    self.pause_event.wait()
                    if self.local_dispatch_count >= self.params.targetDispatches:
                        pending = self.get_next_emergency()
                        if pending:
                            logger.warning("Pending emergency in queue: {}. Waiting for processing.", pending)
                            time.sleep(self.params.poll_interval)
                            continue
                        else:
                            logger.info("Target reached and no pending emergencies.")
                            break
                    with self.active_lock:
                        current = self.active_count
                    if current < self.params.maxActiveCalls:
                        emergency = self.get_next_emergency()
                        if emergency:
                            self.consecutive_no_emergency = 0
                            with self.active_lock:
                                self.active_count += 1
                            pool.submit(self.process_emergency, emergency)
                            logger.info("Submitted emergency at {} {}. Active: {}", emergency.get("city"), emergency.get("county"), self.active_count)
                        else:
                            self.consecutive_no_emergency += 1
                            sleep_time = min(self.params.poll_interval * (2 ** self.consecutive_no_emergency), self.max_poll_sleep)
                            time.sleep(sleep_time)
                    else:
                        time.sleep(self.params.poll_interval)
                    if time.time() - self.last_status_check >= self.params.status_interval:
                        status = self.get_status()
                        if status and status.get("totalDispatches", 0) >= self.params.targetDispatches:
                            logger.info("Remote target reached.")
                            break
                        self.last_status_check = time.time()
        except Exception as e:
            logger.exception("Unhandled exception in simulation run: {}", e)
        finally:
            self.stop_simulation()

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level 5 Simulation")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000")
    parser.add_argument("--seed", type=str, default="default")
    parser.add_argument("--targetDispatches", type=int, default=100)
    parser.add_argument("--maxActiveCalls", type=int, default=15)
    parser.add_argument("--poll_interval", type=float, default=0.3)
    parser.add_argument("--status_interval", type=float, default=5)
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()
    params = SimulationParams(api_url=args.api_url, seed=args.seed,
                              targetDispatches=args.targetDispatches,
                              maxActiveCalls=args.maxActiveCalls,
                              poll_interval=args.poll_interval,
                              status_interval=args.status_interval,
                              debug_mode=args.debug_mode)
    sim = Simulation(params)
    sim.run()

if __name__ == "__main__":
    main()
