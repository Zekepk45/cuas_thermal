#!/usr/bin/env python3
"""
Makes a fake Boson-like thermal video so you can test the whole pipeline
without the drone: sky gradient + sensor noise + warm tree line + sun-warmed
roof (clutter) + a small warm "drone" flying a curved path, with a 0.4 s
video dropout in the middle. Writes truth to <name>_truth.csv.

  python3 make_test_video.py --out test_sky.mp4
  python3 run_tracker.py --source test_sky.mp4 --lens seeker-4.9mm
"""
import argparse
import csv
import math
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="test_sky.mp4")
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--size-px", type=float, default=2.5, help="target size (pixels)")
ap.add_argument("--contrast", type=float, default=45.0, help="target gray levels above sky")
args = ap.parse_args()

W, H = 640, 512
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
sky = 60 + 50 * (yy / H)                                  # sky gets warmer toward horizon
ground_line = 400 + 12 * np.sin(xx / 45.0)               # tree line
scene = np.where(yy > ground_line, 150 + 10 * np.sin(xx / 9.0), sky)
scene[420:470, 80:180] = 200                             # warm roof (static clutter)
fixed_pattern = rng.normal(0, 1.5, (H, W))               # Boson-like FPN

n = int(args.seconds * args.fps)
vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H), isColor=False)
truth = open(args.out.rsplit(".", 1)[0] + "_truth.csv", "w", newline="")
tw = csv.writer(truth)
tw.writerow(["frame", "t", "u", "v", "visible"])

drop = range(int(5.0 * args.fps), int(5.4 * args.fps))  # target hidden 0.4 s
for i in range(n):
    t = i / args.fps
    u = 60 + 520 * (t / args.seconds)
    v = 250 - 120 * math.sin(math.pi * t / args.seconds) + 6 * math.sin(2 * math.pi * 0.7 * t)
    img = scene + fixed_pattern + rng.normal(0, 2.0, (H, W))
    visible = i not in drop
    if visible:
        s = args.size_px / 2.355                          # FWHM -> sigma
        blob = args.contrast * np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * s * s))
        img = img + blob
    frame = np.clip(img, 0, 255).astype(np.uint8)
    vw.write(frame)
    tw.writerow([i, f"{t:.4f}", f"{u:.3f}", f"{v:.3f}", int(visible)])
vw.release()
truth.close()
print(f"wrote {args.out} ({n} frames) and truth csv")
