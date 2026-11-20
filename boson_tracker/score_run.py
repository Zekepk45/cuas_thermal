#!/usr/bin/env python3
"""
Scores a tracker log against truth (synthetic video truth CSV, or later a
truth track from the RTK test beacon projected into the image).

  python3 score_run.py logs/thermal_....csv test_sky_truth.csv
"""
import csv
import math
import sys

log = list(csv.DictReader(open(sys.argv[1])))
truth = {int(r["frame"]): r for r in csv.DictReader(open(sys.argv[2]))}

errs, n_vis, n_valid_vis, n_false = [], 0, 0, 0
first_valid = None
for r in log:
    i = int(r["seq"])
    tr = truth.get(i)
    if tr is None:
        continue
    vis = tr["visible"] == "1"
    valid = r["valid"] == "True"
    n_vis += vis
    if valid:
        e = math.hypot(float(r["px_u"]) - float(tr["u"]), float(r["px_v"]) - float(tr["v"]))
        if e > 5:
            n_false += 1
        else:
            errs.append(e)
            if first_valid is None:
                first_valid = i
        n_valid_vis += vis
errs.sort()
rms = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")
print(f"frames with target visible : {n_vis}")
print(f"valid track on target      : {len(errs)}  ({100*len(errs)/max(n_vis,1):.1f}% of visible frames)")
print(f"valid but WRONG (>5 px off): {n_false}")
print(f"first valid frame          : {first_valid}")
if errs:
    print(f"centroid error px  median {errs[len(errs)//2]:.2f}  95% {errs[int(.95*len(errs))-1]:.2f}  rms {rms:.2f}")
procs = [float(r["proc_ms"]) for r in log if r["proc_ms"]]
print(f"processing time ms  mean {sum(procs)/len(procs):.1f}  max {max(procs):.1f}")
