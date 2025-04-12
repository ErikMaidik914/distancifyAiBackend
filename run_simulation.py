#!/usr/bin/env python3
import argparse
import subprocess
import requests
import json
import os
import time
import math

def run_simulation(script, seed, targetDispatches, maxActiveCalls, api_url):
    cmd = ["python", script, "--api_url", api_url, "--seed", seed, "--targetDispatches", str(targetDispatches), "--maxActiveCalls", str(maxActiveCalls)]
    subprocess.run(cmd, check=True)

def fetch_stats(api_url):
    r = requests.get(f"{api_url}/control/status")
    r.raise_for_status()
    return r.json()

def format_running_time(rt):
    total_seconds = rt.get("totalSeconds", 0) + rt.get("totalNanoseconds", 0) / 1e9
    seconds_part = int(total_seconds)
    frac = total_seconds - seconds_part
    frac_str = f"{int(round(frac * 1e7)):07d}"
    return time.strftime("%H:%M:%S", time.gmtime(seconds_part)) + "." + frac_str

def format_status(status):
    rt = status.get("runningTime", {})
    running_time = format_running_time(rt) if isinstance(rt, dict) else rt
    errors = status.get("errors", {})
    new_status = {
        "status": status.get("status", ""),
        "runningTime": running_time,
        "seed": status.get("seed", ""),
        "requestCount": status.get("requestCount", 0),
        "maxActiveCalls": status.get("maxActiveCalls", 0),
        "totalDispatches": status.get("totalDispatches", 0),
        "targetDispatches": status.get("targetDispatches", 0),
        "distance": status.get("distance", 0),
        "penalty": status.get("penalty", 0),
        "httpRequests": status.get("httpRequests", 0),
        "emulatorVersion": status.get("emulatorVersion", 0),
        "signature": "",
        "checksum": status.get("checksum", ""),
        "errors": {
            "missed": errors.get("missed", 0),
            "overDispatched": errors.get("overDispatched", 0)
        }
    }
    signature = "|".join([
        new_status["status"],
        new_status["runningTime"],
        new_status["seed"],
        str(new_status["requestCount"]),
        str(new_status["maxActiveCalls"]),
        str(new_status["totalDispatches"]),
        str(new_status["targetDispatches"]),
        str(new_status["distance"]),
        str(new_status["penalty"]),
        str(new_status["httpRequests"]),
        str(new_status["emulatorVersion"]),
        str(new_status["errors"]["missed"]),
        str(new_status["errors"]["overDispatched"])
    ])
    new_status["signature"] = signature
    return new_status

def save_stats(folder, level, seed, stats):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"level{level}_{seed}.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run simulation configs and save stats.")
    parser.add_argument("--level", type=str, choices=["1", "2", "3", "4", "5"], required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:5000")
    parser.add_argument("--stats_folder", type=str, default="stats")
    args = parser.parse_args()

    if args.level in ["1", "2", "3"]:
        configs = [
            {"seed": "revolutionrace", "targetDispatches": 10000, "maxActiveCalls": 100},
            {"seed": "jollyroom", "targetDispatches": 100000, "maxActiveCalls": 1000},
            {"seed": "jaktia", "targetDispatches": 10000, "maxActiveCalls": 3},
            {"seed": "bellalite", "targetDispatches": 10000, "maxActiveCalls": 10000}
        ]
    else:
        configs = [{"seed": "gudrun", "targetDispatches": 25, "maxActiveCalls": 5}]

    script = f"level{args.level}.py"

    for cfg in configs:
        run_simulation(script, cfg["seed"], cfg["targetDispatches"], cfg["maxActiveCalls"], args.api_url)
        time.sleep(1)
        stats = fetch_stats(args.api_url)
        formatted = format_status(stats)
        save_stats(args.stats_folder, args.level, cfg["seed"], formatted)

if __name__ == "__main__":
    main()
