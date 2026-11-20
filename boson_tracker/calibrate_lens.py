#!/usr/bin/env python3
"""
Measure the Boson lens distortion with a THERMAL checkerboard and save it for
the tracker (run_tracker.py --calib lens_cal.json).

Making a checkerboard the Boson can see
---------------------------------------
Thermal cameras see temperature/emissivity, not ink. Easiest options:
  - Print a checkerboard on paper, glue it to a thin aluminum sheet or
    cardboard, and heat it with a hair dryer or leave it in the sun for a few
    minutes. Black ink heats faster than white paper, so squares show up.
  - Or tape squares of aluminum foil tape onto cardboard (foil looks "cold",
    cardboard looks "warm" indoors).
Count INNER corners (where 4 squares meet): a board of 10 x 7 squares has
9 x 6 inner corners. Keep it flat.

How to run
----------
  python3 calibrate_lens.py --source 2 --cols 9 --rows 6 --square-mm 40
  python3 calibrate_lens.py --source rtsp://127.0.0.1:8900/live --cols 9 --rows 6 --square-mm 40

  SPACE = keep this view (only when corners are drawn in color)
  c     = calibrate with the views kept so far and save
  q     = quit

Collect 15-30 views. Cover the WHOLE image, especially the corners and edges
(that is where fisheye warping is strongest), and tilt the board in
different directions. Aim for "rms" under 0.5 px.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


def open_cap(src):
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    if src.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(src)


def find_corners(gray, pattern):
    """Try both polarities (the board can look hot-on-cold or cold-on-hot)."""
    for img in (gray, 255 - gray):
        eq = cv2.equalizeHist(img)
        ok, corners = cv2.findChessboardCornersSB(
            eq, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners.reshape(-1, 1, 2).astype(np.float64)
    return None


def calibrate(obj_pts, img_pts, size, model):
    if model == "fisheye":
        K = np.eye(3)
        D = np.zeros((4, 1))
        # flag names live in cv2.fisheye on OpenCV 4.x and in cv2 on OpenCV 5.x
        flag = lambda name: getattr(cv2.fisheye, name, None) or getattr(cv2, name)
        flags = flag("CALIB_RECOMPUTE_EXTRINSIC") | flag("CALIB_FIX_SKEW")
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.reshape(1, -1, 3) for o in obj_pts], [i.reshape(1, -1, 2) for i in img_pts],
            size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9))
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(
            [o.astype(np.float32) for o in obj_pts],
            [i.astype(np.float32) for i in img_pts], size, None, None)
    return rms, K, D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="camera number (e.g. 2), rtsp URL, or a video file")
    ap.add_argument("--cols", type=int, required=True, help="inner corners across")
    ap.add_argument("--rows", type=int, required=True, help="inner corners down")
    ap.add_argument("--square-mm", type=float, required=True, help="size of one square (mm)")
    ap.add_argument("--model", default="fisheye", choices=["fisheye", "pinhole"],
                    help="fisheye for the 4.9 mm lens, pinhole for narrow lenses")
    ap.add_argument("--out", default="lens_cal.json")
    ap.add_argument("--auto", action="store_true",
                    help="keep a view automatically every 1.5 s when corners are found (for video files)")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0

    cap = open_cap(args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.source}")
    obj_pts, img_pts, size = [], [], None
    last_auto = 0.0
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    def do_calibrate():
        if len(img_pts) < 8:
            print(f"need at least 8 views, have {len(img_pts)}")
            return
        rms, K, D = calibrate(obj_pts, img_pts, size, args.model)
        out = {"model": args.model, "image_w": size[0], "image_h": size[1],
               "K": K.tolist(), "D": D.ravel().tolist(), "rms_px": float(rms),
               "n_views": len(img_pts), "board": [args.cols, args.rows, args.square_mm],
               "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nCalibrated from {len(img_pts)} views: rms = {rms:.3f} px "
              f"({'good' if rms < 0.5 else 'OK' if rms < 1.0 else 'poor - add views / flatten board'})")
        print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print(f"  D={np.round(D.ravel(), 5).tolist()}")
        print(f"Saved {args.out}.  Use it with:  python3 run_tracker.py ... --calib {args.out}\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            if not args.source.isdigit() and not args.source.startswith("rtsp://"):
                break   # end of file
            continue
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        corners = find_corners(gray, pattern)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            cv2.drawChessboardCorners(vis, pattern, corners.astype(np.float32), True)
        for p in img_pts:   # show coverage so far
            for q in p[::max(1, len(p) // 8)]:
                cv2.circle(vis, (int(q[0, 0]), int(q[0, 1])), 2, (255, 0, 255), -1)
        cv2.putText(vis, f"views {len(img_pts)}  SPACE=keep  c=calibrate  q=quit", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("calibrate", vis)
        k = cv2.waitKey(1) & 0xFF
        now = time.time()
        keep = (k == ord(" ")) or (args.auto and now - last_auto > 1.5)
        if keep and corners is not None:
            obj_pts.append(objp.copy()); img_pts.append(corners); last_auto = now
            print(f"kept view {len(img_pts)}")
        elif k == ord("c"):
            do_calibrate()
        elif k in (ord("q"), 27):
            break
    if args.auto and img_pts:
        do_calibrate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
