"""
Single-target tracker in image space.

Constant-velocity Kalman filter on the pixel position [u, v, du/dt, dv/dt]
plus a simple state machine that matches the sequence diagram's hold/fault
logic: only a CONFIRMED, healthy track is reported as valid.

  SEARCH     no track. Start one on the best detection.
  TENTATIVE  new track; needs M hits out of the last N frames to confirm.
  TRACK      confirmed and updated this frame  -> valid = True
  COAST      confirmed, but no detection this frame; predicting forward.
             Drops back to SEARCH after max_coast_s.

A click on the video (or set_hint()) tells the tracker which hot spot to
take when several are in view.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np

SEARCH, TENTATIVE, TRACK, COAST = "SEARCH", "TENTATIVE", "TRACK", "COAST"


@dataclass
class TrackerConfig:
    meas_sigma_px: float = 0.7      # centroid noise, 1-sigma (pixels)
    accel_sigma_px: float = 400.0   # process noise, pixels/s^2 (target + camera shake)
    gate_chi2: float = 13.8         # 99.9 % gate for 2 DOF
    min_gate_px: float = 6.0        # never gate tighter than this
    confirm_m: int = 3
    confirm_n: int = 5
    max_coast_s: float = 0.5
    hint_radius_px: float = 40.0


class SingleTargetTracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.reset()

    # ------------------------------------------------------------------ api
    def reset(self):
        self.state = SEARCH
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.t = None
        self.t_last_hit = None
        self.hits = deque(maxlen=self.cfg.confirm_n)
        self.track_id = getattr(self, "track_id", 0)
        self.last_det = None
        self.hint = getattr(self, "hint", None)

    def set_hint(self, u, v):
        """Operator click: prefer the hot spot nearest this pixel."""
        self.hint = (float(u), float(v))
        self.reset()
        self.hint = (float(u), float(v))

    def clear_hint(self):
        self.hint = None

    @property
    def valid(self):
        return self.state == TRACK

    def position(self):
        return float(self.x[0]), float(self.x[1])

    def velocity(self):
        return float(self.x[2]), float(self.x[3])

    def pos_sigma_px(self):
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))

    def step(self, t: float, detections):
        """Advance to time t (seconds) with this frame's detections."""
        c = self.cfg
        self.last_det = None

        if self.state == SEARCH:
            d = self._pick_new(detections)
            if d is not None:
                self._start(t, d)
            return self.state

        self._predict(t)
        d = self._associate(detections)
        if d is not None:
            self._update(d)
            self.t_last_hit = t
            self.hits.append(1)
            self.last_det = d
        else:
            self.hits.append(0)

        if self.state == TENTATIVE:
            if sum(self.hits) >= c.confirm_m:
                self.state = TRACK
                self.hint = None
            elif len(self.hits) == c.confirm_n:
                self.reset()
        elif self.state in (TRACK, COAST):
            if d is not None:
                self.state = TRACK
            elif t - self.t_last_hit > c.max_coast_s:
                self.reset()
            else:
                self.state = COAST
        return self.state

    # ------------------------------------------------------------- internals
    def _pick_new(self, dets):
        if not dets:
            return None
        if self.hint is not None:
            hu, hv = self.hint
            near = [d for d in dets
                    if (d.u - hu) ** 2 + (d.v - hv) ** 2 <= self.cfg.hint_radius_px ** 2]
            if not near:
                return None
            return min(near, key=lambda d: (d.u - hu) ** 2 + (d.v - hv) ** 2)
        return dets[0]  # highest SNR

    def _start(self, t, d):
        self.track_id += 1
        self.state = TENTATIVE
        self.x = np.array([d.u, d.v, 0.0, 0.0])
        r = self.cfg.meas_sigma_px ** 2
        self.P = np.diag([r, r, 200.0 ** 2, 200.0 ** 2])
        self.t = t
        self.t_last_hit = t
        self.hits.clear()
        self.hits.append(1)
        self.last_det = d

    def _predict(self, t):
        dt = max(1e-3, min(t - self.t, 1.0))
        self.t = t
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = self.cfg.accel_sigma_px ** 2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ (q * np.eye(2)) @ G.T

    def _associate(self, dets):
        if not dets:
            return None
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.cfg.meas_sigma_px ** 2
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        best, best_cost = None, None
        for d in dets:
            y = np.array([d.u, d.v]) - self.x[:2]
            m2 = float(y @ Si @ y)
            ok = m2 <= self.cfg.gate_chi2 or float(y @ y) <= self.cfg.min_gate_px ** 2
            if not ok:
                continue
            cost = m2 - 0.1 * d.snr  # prefer close and strong
            if best_cost is None or cost < best_cost:
                best, best_cost = d, cost
        return best

    def _update(self, d):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.cfg.meas_sigma_px ** 2
        y = np.array([d.u, d.v]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
