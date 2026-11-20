"""
Outputs from the thermal processor (A4):
  - UDP JSON "thermal_los" messages to the fusion node (A6), same transport
    style as last year's SkyDome radar feed (UDP JSON), on its own port.
  - CSV log of every frame for the data logger (DL1) / KPP scoring.
"""
import csv
import json
import os
import socket
from datetime import datetime, timezone

SCHEMA_VERSION = 1


def iso_utc(t_unix: float) -> str:
    return datetime.fromtimestamp(t_unix, tz=timezone.utc).isoformat(
        timespec="microseconds").replace("+00:00", "Z")


def build_message(seq, t_unix, time_source, sensor_id, lens_key, cam, trk,
                  frame_wh, fps, fault, rng=None):
    """One thermal line-of-sight measurement. Sent every frame, valid or not,
    so fusion always knows the sensor's status (hold / fault path)."""
    msg = {
        "msg_type": "thermal_los",
        "schema": SCHEMA_VERSION,
        "sensor_id": sensor_id,
        "seq": seq,
        "t_unix": round(t_unix, 6),
        "t_iso": iso_utc(t_unix),
        "time_source": time_source,
        "track_state": trk.state,
        "track_id": trk.track_id if trk.state != "SEARCH" else None,
        "valid": bool(trk.valid and fault is None),
        "fault": fault,
        "lens": lens_key,
        "frame_wh": list(frame_wh),
        "fps": round(fps, 1),
    }
    if trk.state != "SEARCH":
        u, v = trk.position()
        du, dv = trk.velocity()
        az, el, los = cam.pixel_to_los(u, v)
        sig_px = trk.pos_sigma_px()
        az_rate, el_rate = cam.angle_rates_dps(u, v, du, dv)   # distortion included
        msg.update({
            "az_deg": round(az, 5),
            "el_deg": round(el, 5),
            "los_frd": [round(c, 6) for c in los],
            "sigma_deg": round(cam.pixel_sigma_to_deg(sig_px, u, v), 5),
            "az_rate_dps": round(az_rate, 4),
            "el_rate_dps": round(el_rate, 4),
            "px": [round(u, 2), round(v, 2)],
            "lens_calibrated": cam.calibrated,
        })
        d = trk.last_det
        if d is not None:
            msg.update({"snr": round(d.snr, 1), "area_px": d.area})
        if rng is not None:
            # range from apparent size (see range_estimator.py). range_m is None
            # when the target is too small to measure; then range_min_m is a lower bound.
            msg.update(rng)
    return msg


class UdpJsonSender:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if host.endswith(".255"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, msg: dict):
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass  # never let a network hiccup stop tracking


CSV_FIELDS = ["seq", "t_unix", "t_iso", "time_source", "track_state", "track_id",
              "valid", "fault", "az_deg", "el_deg", "sigma_deg", "az_rate_dps",
              "el_rate_dps", "px_u", "px_v", "snr", "area_px",
              "range_m", "range_sigma_m", "range_min_m", "range_status",
              "size_px", "size_true_px", "ang_size_mrad",
              "fps", "cam_fps", "dropped", "lag_ms", "n_detections", "proc_ms"]


class CsvLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self.w.writeheader()
        self.path = path

    def log(self, msg: dict, n_dets: int, proc_ms: float, extra: dict = None):
        row = dict(msg)
        if extra:
            row.update(extra)
        px = msg.get("px")
        if px:
            row["px_u"], row["px_v"] = px
        row["n_detections"] = n_dets
        row["proc_ms"] = round(proc_ms, 2)
        self.w.writerow(row)

    def close(self):
        self.f.close()
