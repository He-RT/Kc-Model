"""Collect CPU/GPU/RAM stats and write to JSON. Run on mlpc via nohup."""

import json, subprocess, time, os
from pathlib import Path

OUT = Path.home() / "stats.json"

while True:
    stats = {"time": time.strftime("%H:%M:%S")}

    # CPU + RAM
    try:
        r = subprocess.run(["free", "-h"], capture_output=True, text=True)
        lines = r.stdout.strip().split("\n")
        stats["ram"] = lines[1].split()[1:]  # total, used, free
    except: stats["ram"] = ["N/A"]*3

    # GPU
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        g = r.stdout.strip().split(", ")
        stats["gpu"] = {"util": g[0], "mem_used": g[1], "mem_total": g[2], "temp": g[3], "power": g[4]}
    except: stats["gpu"] = "N/A"

    # Training process
    try:
        r = subprocess.run(["ps", "-eo", "pid,%cpu,%mem,etime,comm"], capture_output=True, text=True)
        procs = []
        for line in r.stdout.split("\n")[1:]:
            if "python" in line and "stats" not in line:
                parts = line.split()
                procs.append({"pid": parts[0], "cpu": parts[1], "mem": parts[2], "time": parts[3], "cmd": " ".join(parts[4:])[:60]})
        stats["processes"] = procs[:5]
    except: stats["processes"] = []

    OUT.write_text(json.dumps(stats))
    time.sleep(2)
