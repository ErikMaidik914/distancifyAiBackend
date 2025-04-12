#!/usr/bin/env python3
import os
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from scipy.spatial import KDTree
import argparse

SESSION = None
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()

def pause_simulation():
    PAUSE_EVENT.clear()
    logger.info("Simulation paused")

def resume_simulation():
    PAUSE_EVENT.set()
    logger.info("Simulation resumed")

def stop_simulation():
    STOP_EVENT.set()
    logger.info("Simulation stop triggered")

class SimulationParams:
    def __init__(self, api_url=None, seed="default", targetDispatches=100,
                 maxActiveCalls=15, poll_interval=0.3, status_interval=5):
        self.api_url = api_url or os.environ.get("API_BASE_URL", "http://localhost:5000")
        self.seed = seed
        self.targetDispatches = targetDispatches
        self.maxActiveCalls = maxActiveCalls
        self.poll_interval = poll_interval
        self.status_interval = status_interval

def create_session(base_url, retries=3, backoff=0.5):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500,502,503,504])
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.base_url = base_url
    logger.info(f"HTTP session created for base url {base_url}")
    return s

def _initialize_supply(session):
    url = f"{session.base_url}/medical/search"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    supply_data = {}
    points, keys = [], []
    for d in data:
        k = (d["county"], d["city"])
        supply_data[k] = {"quantity": d["quantity"], "lat": d["latitude"], "lon": d["longitude"]}
        keys.append(k)
        points.append((d["latitude"], d["longitude"]))
    logger.info("Supply initialization complete")
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
    if resp.ok:
        logger.info(f"Dispatched {qty} from {srcCounty}/{srcCity} to {tgtCounty}/{tgtCity}")
    else:
        logger.error(f"Dispatch failed from {srcCounty}/{srcCity} to {tgtCounty}/{tgtCity}: {resp.status_code}")
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
    logger.info("Simulation starting")
    local_supply = {}
    tree = None
    supply_keys = []
    active_count = 0
    local_dispatch_count = 0
    reset_url = f"{SESSION.base_url}/control/reset?seed={params.seed}&targetDispatches={params.targetDispatches}&maxActiveCalls={params.maxActiveCalls}"
    r = SESSION.post(reset_url)
    if not r.ok:
        logger.error(f"Reset call failed: {r.status_code} {r.text}")
        return
    logger.info("Backend simulation reset complete")
    try:
        local_supply, tree_obj, supply_keys = _initialize_supply(SESSION)
    except Exception as e:
        logger.error(f"Supply initialization failed: {e}")
        return
    tree = tree_obj
    lock = threading.Lock()
    def process_emergency(call):
        nonlocal active_count, local_dispatch_count
        needed = sum(x["Quantity"] for x in call.get("requests", []))
        if needed <= 0:
            with lock:
                active_count -= 1
            logger.info("No emergency resources needed")
            return
        pt = (call["latitude"], call["longitude"])
        _, idxs = tree.query(pt, k=len(supply_keys))
        remain = needed
        with lock:
            for i in idxs:
                key = supply_keys[i]
                av = local_supply[key]["quantity"]
                if av <= 0:
                    continue
                use = min(av, remain)
                if _dispatch(SESSION, key[0], key[1], call["county"], call["city"], use):
                    local_supply[key]["quantity"] -= use
                    remain -= use
                    local_dispatch_count += use
                if remain <= 0:
                    break
            active_count -= 1
        logger.info(f"Processed emergency call at {call['county']}/{call['city']} with required {needed} fulfilled by dispatching {needed - remain}")
    threads = []
    last_status_check = time.time()
    while True:
        if STOP_EVENT.is_set():
            logger.info("Stop event detected")
            break
        PAUSE_EVENT.wait()
        if local_dispatch_count >= params.targetDispatches:
            logger.info("Target dispatch count reached")
            break
        if time.time() - last_status_check >= params.status_interval:
            st = _get_status(SESSION)
            if st and st.get("totalDispatches", 0) >= params.targetDispatches:
                logger.info("Backend reported target dispatch count reached")
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
                logger.info("Started emergency processing thread")
            else:
                time.sleep(params.poll_interval)
        else:
            time.sleep(params.poll_interval)
        threads = [t for t in threads if t.is_alive()]
    stop = SESSION.post(f"{SESSION.base_url}/control/stop")
    if not stop.ok:
        logger.error(f"Stop call failed: {stop.status_code} {stop.text}")
    logger.info("Notified backend simulation end")
    for t in threads:
        t.join()
    STOP_EVENT.clear()
    logger.info("Simulation finished")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Level 1 Simulation")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000")
    parser.add_argument("--seed", type=str, default="default")
    parser.add_argument("--targetDispatches", type=int, default=100)
    parser.add_argument("--maxActiveCalls", type=int, default=15)
    parser.add_argument("--poll_interval", type=float, default=0.3)
    parser.add_argument("--status_interval", type=float, default=5)
    args = parser.parse_args()
    params = SimulationParams(api_url=args.api_url, seed=args.seed,
                              targetDispatches=args.targetDispatches,
                              maxActiveCalls=args.maxActiveCalls,
                              poll_interval=args.poll_interval,
                              status_interval=args.status_interval)
    run_simulation(params)
