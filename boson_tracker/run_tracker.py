#!/usr/bin/env python3
"""
Team Thermal - Boson thermal tracker (block A4, Thermal Processing).

  grab frame -> find hot target -> centroid -> Kalman track -> az/el angle
  (fisheye corrected) -> range from apparent size -> timestamp + health check
  -> UDP JSON to fusion (A6) + CSV log (DL1) + on-screen and console readout

Sources:
  --source rtsp://127.0.0.1:8900/live   Seeker drone via voxl-streamer + adb forward
  --source 2                            USB camera number 2 (= /dev/video2). Use the NUMBER.
  --source flight.mp4                   recorded video

Keys (video window):  click = lock onto that hot spot   r = reset track
                      c = clear lock hint   s = save snapshot   q / Esc = quit
"""
# =============================================================================
#  USER SETTINGS  -  edit these.  (Command-line options, if given, override them.)
# =============================================================================
# --- Drone size, for range-from-size -----------------------------------------
DRONE_SIZE_M = 0.35        # <<< ENTER YOUR DRONE'S SIZE HERE (meters) >>>
                           # Start with the longest dimension that shows up warm in the
                           # thermal image (usually motor-to-motor diagonal; the cold props
                           # barely show). Then CALIBRATE it (best): put the drone, powered
                           # and warmed up, at a measured distance (tape / laser, 20-40 m)
                           # and run with --cal-range <that distance>. The screen and console
                           # show "set DRONE_SIZE_M = x.xxx". Put that number here.
DRONE_SIZE_UNCERT = 0.25   # 1-sigma uncertainty of DRONE_SIZE_M as a fraction (0.25 = 25 %).
                           # Covers viewing angle (side-on looks smaller than diagonal).
                           # After a --cal-range calibration, use 0.10.
# --- Image blur / measurement noise -----------------------------------------
PSF_PX = 2.5               # Size (px) that a POINT-like hot spot measures (the blur).
                           # MEASURE IT on your stream: aim at a small hot object far away
                           # (e.g. a lighter or soldering-iron tip 20+ m off) and read
                           # "size ... px" on screen. Put that number here.
PSF_SIGMA_PX = 0.3         # Uncertainty of PSF_PX (px).
SIZE_SIGMA_PX = 0.3        # Single-frame noise of the size measurement (px).
MIN_RESOLVED_PX = 1.5      # True size (px, blur removed) below which only "range > X" is given.
MIN_SNR_FOR_RANGE = 30.0   # Below this SNR the size reads too big, so only "range > X" is given.
RANGE_SMOOTH_S = 0.5       # Average the size over this many seconds (0 = no averaging).
# --- Lens ---------------------------------------------------------------------
LENS_CALIBRATION = None    # e.g. "lens_cal.json" from calibrate_lens.py. None = nominal
                           # 95 deg fisheye model from FLIR's spec.
# --- Display / console / recording ------------------------------------------
CONSOLE_HZ = 10            # Console lines per second. 0 = one line for EVERY frame.
SHOW_LOCAL_TIME = True     # Also show local (Mountain) time next to UTC on the video.
RECORD_FPS = 30.0          # Frame rate of --record files. Frames are written on the
                           # real clock, so playback is always real time.
# =============================================================================

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from camera_model import PROFILES, CameraModel
from detector import DetectorConfig, HotSpotDetector
from tracker import TrackerConfig, SingleTargetTracker, TRACK, TENTATIVE, COAST
from output import UdpJsonSender, CsvLogger, build_message
from range_estimator import RangeConfig, RangeEstimator


# --------------------------------------------------------------------------- video
class LatestFrameGrabber:
    """Reads a live stream on its own thread and keeps only the newest frame,
    so tracking never falls behind real time. If processing is slower than the
    camera, older frames are DROPPED (counted), never queued up - queued frames
    are what makes live video look like slow motion."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame, self.t, self.n = None, 0.0, 0
        self.cam_fps = 0.0
        self.ok = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        t_last = None
        while self.ok:
            ok, f = self.cap.read()
            t = time.time()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.t, self.n = f, t, self.n + 1
                if t_last is not None and t > t_last:
                    inst = 1.0 / (t - t_last)
                    self.cam_fps = inst if self.cam_fps == 0 else 0.95 * self.cam_fps + 0.05 * inst
            t_last = t

    def read(self, last_n, timeout=1.0):
        """Wait for a frame newer than last_n. Returns (frame, t, n) or None."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.n != last_n:
                    return self.frame, self.t, self.n
            time.sleep(0.001)
        return None

    def stop(self):
        self.ok = False


def open_source(src, no_low_latency=False):
    if src.startswith("/dev/video") and src[10:].isdigit():
        src = src[10:]            # "/dev/video2" -> "2" (OpenCV wants the number)
    is_live = src.isdigit() or src.startswith(("rtsp://", "http://", "udp://"))
    # low-latency FFmpeg options only for network streams
    if src.startswith("rtsp://") and not no_low_latency:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if is_live else cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video source '{src}'. "
                 "For the drone: is voxl-streamer running and 'adb forward tcp:8900 tcp:8900' set? "
                 "For USB: check 'v4l2-ctl --list-devices' and pass the number.")
    if is_live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep the driver's queue short (USB cameras)
    return cap, is_live


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- time text
def utc_text(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def local_text(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3] + " local"


def range_text(m):
    st = m.get("range_status")
    if st in ("ok", "held") and m.get("range_m") is not None:
        held = " (held)" if st == "held" else ""
        return f"range {m['range_m']:7.1f} +- {m['range_sigma_m']:.1f} m (1-sigma){held}"
    if st == "unresolved":
        return f"range > {m['range_min_m']:.0f} m (too small to size)"
    if st == "low_snr":
        return f"range > {m['range_min_m']:.0f} m (too faint to size)"
    return "range  --"


# --------------------------------------------------------------------------- display
COLORS = {TRACK: (0, 255, 0), TENTATIVE: (0, 255, 255), COAST: (0, 165, 255)}


def put(vis, s, x, y, color=(0, 255, 0), scale=0.5):
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame_gray, dets, trk, cam, msg, stats, hint, t_frame, cal_text=None):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    cx, cy = int(round(cam.cx)), int(round(cam.cy))
    cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for d in dets:
        cv2.circle(vis, (int(d.u), int(d.v)), 6, (180, 180, 180), 1)
    if hint is not None:
        cv2.circle(vis, (int(hint[0]), int(hint[1])), 40, (255, 0, 255), 1)
    if trk.state in COLORS:
        u, v = trk.position()
        col = COLORS[trk.state]
        s = 14
        cv2.rectangle(vis, (int(u) - s, int(v) - s), (int(u) + s, int(v) + s), col, 2)
        du, dv = trk.velocity()
        cv2.line(vis, (int(u), int(v)), (int(u + du * 0.3), int(v + dv * 0.3)), col, 1)
        cv2.line(vis, (cx, cy), (int(u), int(v)), col, 1, cv2.LINE_AA)
        if msg.get("range_m") is not None:   # range label next to the box
            put(vis, f"{msg['range_m']:.0f}+-{msg['range_sigma_m']:.0f} m", int(u) + s + 4, int(v), col, 0.45)

    lines = [f"{trk.state}  id={msg.get('track_id')}  valid={msg['valid']}",
             f"dets {len(dets)}"]
    if "az_deg" in msg:
        lines.append(f"az {msg['az_deg']:+8.3f} deg  el {msg['el_deg']:+8.3f} deg  "
                     f"(+-{msg['sigma_deg']:.3f})")
        lines.append(range_text(msg))
        if msg.get("size_px") is not None:
            tp = msg.get("size_true_px")
            lines.append(f"size {msg['size_px']:.1f} px  (deblurred {tp if tp is not None else '--'})  "
                         f"snr {msg.get('snr', 0):.0f}")
    if cal_text:
        lines.append(cal_text)
    if msg.get("fault"):
        lines.append(f"FAULT: {msg['fault']}")
    for i, s in enumerate(lines):
        put(vis, s, 6, 18 + 18 * i)

    # running timestamp (bottom-left) and real-time health (bottom-right)
    ts = utc_text(t_frame) + (f"   {local_text(t_frame)}" if SHOW_LOCAL_TIME else "")
    put(vis, ts, 6, h - 26, (255, 255, 255))
    put(vis, f"frame {msg['seq']}", 6, h - 8, (255, 255, 255))
    health = (f"cam {stats['cam_fps']:4.1f} fps  proc {stats['fps']:4.1f} fps  "
              f"lag {stats['lag_ms']:4.0f} ms  drop {stats['dropped']}")
    col = (0, 255, 0) if stats["dropped_recent"] == 0 else (0, 165, 255)
    (tw, _), _ = cv2.getTextSize(health, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    put(vis, health, w - tw - 6, h - 8, col, 0.45)
    if not cam.calibrated:
        put(vis, "lens: nominal model", w - 160, 18, (0, 200, 255), 0.45)
    return vis


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="rtsp://127.0.0.1:8900/live")
    ap.add_argument("--lens", default="seeker-4.9mm", choices=PROFILES.keys())
    ap.add_argument("--calib", default=LENS_CALIBRATION, help="lens_cal.json from calibrate_lens.py")
    ap.add_argument("--sensor-id", default="BOSON-SEEKER")
    ap.add_argument("--udp-host", default="127.0.0.1", help="fusion node IP")
    ap.add_argument("--udp-port", type=int, default=9101,
                    help="thermal LOS port (radar SkyDome feed uses 9001)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--record", default=None, help="save annotated video, e.g. run1.mp4")
    ap.add_argument("--no-display", action="store_true", help="headless (field box / test)")
    ap.add_argument("--fast", action="store_true",
                    help="video files: process as fast as possible instead of real time")
    ap.add_argument("--no-low-latency", action="store_true",
                    help="turn off low-latency RTSP decode options (try this if the image looks garbled)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    ap.add_argument("--console-hz", type=float, default=CONSOLE_HZ,
                    help="console lines per second (0 = every frame)")
    # range from size
    ap.add_argument("--drone-size", type=float, default=DRONE_SIZE_M, help="drone size seen by the camera (m)")
    ap.add_argument("--size-uncert", type=float, default=DRONE_SIZE_UNCERT, help="1-sigma, fraction")
    ap.add_argument("--psf-px", type=float, default=PSF_PX, help="blur size of a point target (px)")
    ap.add_argument("--cal-range", type=float, default=None,
                    help="CALIBRATION: drone is held at this known range (m); prints the "
                         "DRONE_SIZE_M value that makes range come out right")
    # detector tuning
    ap.add_argument("--mode", default="local", choices=["local", "global", "hottest"],
                    help="local = small/distant targets; global = big close targets (bench); "
                         "hottest = only the single hottest object in view")
    ap.add_argument("--hot-delta", type=float, default=12.0,
                    help="hottest mode: how far below the hottest pixel (gray levels) still counts as the object")
    ap.add_argument("--k-sigma", type=float, default=5.0, help="detection threshold (higher = fewer false hits)")
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--max-area", type=int, default=5000)
    ap.add_argument("--bg-kernel", type=int, default=15,
                    help="local mode: biggest target size in pixels it will keep")
    ap.add_argument("--static-tau", type=float, default=4.0,
                    help="seconds before a still hot spot is treated as clutter; 0 = off (use 0 on a moving gimbal)")
    ap.add_argument("--black-hot", action="store_true")
    # health
    ap.add_argument("--max-gap-s", type=float, default=0.25,
                    help="frame gap that counts as a dropped-video fault")
    args = ap.parse_args()

    cap, is_live = open_source(args.source, args.no_low_latency)
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grabber = LatestFrameGrabber(cap) if is_live else None

    # first frame -> size -> camera model
    if grabber:
        first = grabber.read(-1, timeout=10.0)
        if first is None:
            sys.exit("ERROR: stream opened but no frames arrived in 10 s.")
        frame, t_frame, n_frame = first
    else:
        ok, frame = cap.read()
        if not ok:
            sys.exit("ERROR: could not read first frame.")
        t_start = time.time()
        n_frame, t_frame = 0, t_start
    gray = to_gray(frame)
    h, w = gray.shape[:2]
    cam = CameraModel(PROFILES[args.lens], w, h, args.calib)
    print(cam.summary())
    for r in (50, 100, 300):
        print(f"  {args.drone_size:.2f} m drone at {r:3d} m = "
              f"{cam.pixels_on_target(args.drone_size, r):5.1f} px wide (image center)")

    det = HotSpotDetector(DetectorConfig(
        mode=args.mode, k_sigma=args.k_sigma, min_area=args.min_area,
        max_area=args.max_area, bg_kernel=args.bg_kernel, black_hot=args.black_hot,
        temporal_tau_s=0.0 if args.mode == "hottest" else args.static_tau,
        hot_delta=args.hot_delta))
    trk = SingleTargetTracker(TrackerConfig())
    rng_est = RangeEstimator(RangeConfig(
        target_size_m=args.drone_size, size_uncert_frac=args.size_uncert,
        psf_px=args.psf_px, psf_sigma_px=PSF_SIGMA_PX, size_sigma_px=SIZE_SIGMA_PX,
        min_resolved_px=MIN_RESOLVED_PX, min_snr=MIN_SNR_FOR_RANGE, smooth_tau_s=RANGE_SMOOTH_S))
    print(f"Range from size: drone {args.drone_size} m (+-{100*args.size_uncert:.0f} %), "
          f"blur {args.psf_px} px")
    if args.mode == "local":
        big = args.drone_size / (2 * np.tan(args.bg_kernel * cam.rad_per_px(cam.cx, cam.cy) / 2))
        print(f"  note: --mode local ignores targets wider than {args.bg_kernel} px, "
              f"so range closer than ~{big:.0f} m reads too far. Use --mode global/hottest up close.")
    udp = UdpJsonSender(args.udp_host, args.udp_port)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = CsvLogger(os.path.join(args.log_dir, f"thermal_{args.sensor_id}_{stamp}.csv"))
    print(f"Sending thermal_los JSON to udp://{args.udp_host}:{args.udp_port}")
    print(f"Logging to {log.path}")

    writer, rec_t0, rec_written = None, None, 0
    rec_fps = RECORD_FPS if is_live else file_fps
    win = "Team Thermal - Boson tracker"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, w * 2 if w < 800 else w, h * 2 if w < 800 else h)
        cv2.setMouseCallback(win, lambda e, x, y, f, p: trk.set_hint(x, y)
                             if e == cv2.EVENT_LBUTTONDOWN else None)

    time_source = "host_rx" if is_live else "file_fps"
    realtime_file = (not is_live) and (not args.no_display) and (not args.fast)
    seq, t_prev, fps = 0, None, 0.0
    dropped, drop_hist = 0, []
    last_console, last_warn = 0.0, 0.0
    n_prev = n_frame
    try:
        while True:
            t0 = time.perf_counter()
            fault = None
            if seq > 0:
                if grabber:
                    got = grabber.read(n_frame, timeout=1.0)
                    if got is None:
                        # fault path: no video -> report invalid, keep trying
                        trk.reset(); rng_est.reset()
                        msg = build_message(seq, time.time(), time_source, args.sensor_id,
                                            args.lens, cam, trk, (w, h), 0.0, "no_video")
                        udp.send(msg); log.log(msg, 0, 0.0); seq += 1
                        print(f"[{utc_text(time.time())}] WARN: no video for 1 s")
                        continue
                    frame, t_frame, n_frame = got
                else:
                    if realtime_file:            # play files at real speed, not as fast as possible
                        wait = t_start + (n_frame + 1) / file_fps - time.time()
                        if wait > 0:
                            time.sleep(wait)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    n_frame += 1
                    t_frame = t_start + n_frame / file_fps
                gray = to_gray(frame)
                t0 = time.perf_counter()

            # frames the camera sent that we skipped because processing was busy
            skipped = max(0, n_frame - n_prev - 1) if grabber else 0
            n_prev = n_frame
            dropped += skipped
            drop_hist = (drop_hist + [skipped])[-60:]

            # health: timestamp sanity and frame gaps
            if t_prev is not None:
                dt = t_frame - t_prev
                if dt <= 0:
                    fault = "timestamp_not_increasing"
                elif dt > args.max_gap_s:
                    fault = f"frame_gap_{dt:.2f}s"
                inst = 1.0 / dt if dt > 0 else 0.0
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            t_prev = t_frame

            protect = None
            if trk.state != "SEARCH":
                pu, pv = trk.position()
                protect = (pu, pv, 12)
            dets = det.detect(gray, t_frame, protect)
            trk.step(t_frame, dets)

            rng = None
            if trk.state in (TRACK, COAST):
                u, v = trk.position()
                rng = rng_est.update(t_frame, trk.last_det, cam, u, v, trk.track_id)
            elif trk.state == "SEARCH":
                rng_est.reset()

            msg = build_message(seq, t_frame, time_source, args.sensor_id, args.lens,
                                cam, trk, (w, h), fps, fault, rng)
            udp.send(msg)
            proc_ms = (time.perf_counter() - t0) * 1000
            lag_ms = (time.time() - t_frame) * 1000 if is_live else proc_ms
            stats = {"fps": fps, "cam_fps": grabber.cam_fps if grabber else file_fps,
                     "lag_ms": lag_ms, "dropped": dropped, "dropped_recent": sum(drop_hist)}
            log.log(msg, len(dets), proc_ms,
                    {"cam_fps": round(stats["cam_fps"], 1), "dropped": dropped, "lag_ms": round(lag_ms, 1)})

            cal_text = None
            if args.cal_range:
                eff = rng_est.effective_size_at(args.cal_range)
                cal_text = (f"CAL @ {args.cal_range:g} m: set DRONE_SIZE_M = {eff:.3f}"
                            if eff else f"CAL @ {args.cal_range:g} m: waiting for a sized track")

            # ---- console line (with timestamp) ----
            now = time.time()
            # pace console lines by FRAME time, so files and live streams behave the same
            if args.console_hz <= 0 or t_frame - last_console >= 1.0 / args.console_hz:
                last_console = t_frame
                if "az_deg" in msg:
                    body = (f"{trk.state:9s} id={msg['track_id']}  az {msg['az_deg']:+8.3f}  "
                            f"el {msg['el_deg']:+8.3f} (+-{msg['sigma_deg']:.3f}) deg  "
                            f"{range_text(msg)}")
                else:
                    body = f"{trk.state:9s} no target"
                print(f"[{utc_text(t_frame)}] #{seq:6d} {body}  | proc {fps:4.1f} fps"
                      f" lag {lag_ms:4.0f} ms" + (f" | {cal_text}" if cal_text else "")
                      + (f" | FAULT {fault}" if fault else ""))
            if grabber and sum(drop_hist) > 0 and now - last_warn > 5.0:
                last_warn = now
                print(f"[{utc_text(now)}] WARN: processing ({fps:.1f} fps) slower than camera "
                      f"({grabber.cam_fps:.1f} fps): showing newest frames only, {dropped} skipped so far")

            seq += 1

            if not args.no_display or args.record:
                vis = draw(gray, dets, trk, cam, msg, stats, trk.hint, t_frame, cal_text)
                if args.record:
                    # write on the REAL clock: if frames come slower than rec_fps the
                    # last frame is repeated, so playback is always real time
                    if writer is None:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 rec_fps, (vis.shape[1], vis.shape[0]))
                        rec_t0 = t_frame
                        print(f"Recording {args.record} at {rec_fps:g} fps (real-time)")
                    due = int((t_frame - rec_t0) * rec_fps) + 1
                    for _ in range(min(due - rec_written, int(2 * rec_fps))):
                        writer.write(vis)
                    rec_written = max(rec_written, due)
                if not args.no_display:
                    cv2.imshow(win, vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    elif k == ord("r"):
                        trk.reset(); rng_est.reset()
                    elif k == ord("c"):
                        trk.clear_hint(); trk.reset(); rng_est.reset()
                    elif k == ord("s"):
                        fn = f"snap_{time.strftime('%H%M%S')}.png"
                        cv2.imwrite(fn, vis); print("saved", fn)

            if args.max_frames and seq >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if grabber:
            grabber.stop()
        cap.release()
        if writer:
            writer.release()
        log.close()
        cv2.destroyAllWindows()
        print(f"Done. {seq} frames processed, {dropped} skipped. Log: {log.path}")


if __name__ == "__main__":
    main()
