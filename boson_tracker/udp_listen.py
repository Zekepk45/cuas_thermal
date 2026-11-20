#!/usr/bin/env python3
"""
Stand-in for the fusion node (A6): receives thermal_los JSON and prints it,
plus the message rate (KSA-3 needs >= 5 Hz) and age of each message.

  python3 udp_listen.py              # listen on 0.0.0.0:9101
  python3 udp_listen.py --port 9101 --raw
"""
import argparse
import json
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9101)
ap.add_argument("--raw", action="store_true", help="print the full JSON")
args = ap.parse_args()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", args.port))
print(f"Listening for thermal_los on UDP {args.port} ...")

window = []
n_valid = n_total = 0
last_print = 0.0
while True:
    data, addr = s.recvfrom(65535)
    now = time.time()
    m = json.loads(data)
    n_total += 1
    n_valid += bool(m.get("valid"))
    window = [t for t in window if now - t < 1.0] + [now]
    if args.raw:
        print(json.dumps(m))
        continue
    if now - last_print >= 0.2:  # 5 lines/s so it stays readable
        last_print = now
        age_ms = (now - m["t_unix"]) * 1000
        if "az_deg" in m:
            body = f"az {m['az_deg']:+8.3f}  el {m['el_deg']:+8.3f}  sig {m['sigma_deg']:.4f} deg"
        else:
            body = "no target"
        print(f"#{m['seq']:6d} {m['sensor_id']} {m['track_state']:9s} valid={str(m['valid']):5s} "
              f"{body}  rate {len(window):3d} Hz  age {age_ms:6.1f} ms  "
              f"valid {100*n_valid/n_total:5.1f}%  {m.get('fault') or ''}")
