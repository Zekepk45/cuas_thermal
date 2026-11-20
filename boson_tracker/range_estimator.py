"""
Range from apparent size (block A4 add-on).

Idea: if you know how big the drone really is (S, meters) and you measure how
big it looks (alpha, radians), then

        range = S / (2 * tan(alpha / 2))   ~=  S / alpha

alpha comes from the detector's size measurement (pixels) times how much angle
one pixel covers at that spot in the image (camera_model.rad_per_px, which
includes the fisheye warping).

Two corrections make the pixel size honest:
  1. Blur (PSF). Even a point-sized hot spot looks ~1-2 px wide because of
     the lens, the pixel size and video compression. That blur is removed
     "in quadrature":   true_px = sqrt(measured_px^2 - psf_px^2)
  2. Smoothing. One frame's size is noisy (a 3 px blob can only be 2, 3 or 4
     px). The size is averaged over ~smooth_tau_s seconds before range is
     computed.

Error (1-sigma) combines two independent parts:
        sigma_R / R = sqrt( (sigma_px / true_px)^2 + size_uncert_frac^2 )
  - sigma_px / true_px : how well the size in pixels is measured
  - size_uncert_frac   : how well you know the drone's size AS SEEN BY THE
                         THERMAL CAMERA (viewing angle, cold props, hot motors)

When the target is too small (true size < min_resolved_px) or too faint
(SNR < min_snr) to size well, the estimator only reports a LOWER bound:
"range is at least R_min". (Tested: with SNR >= 30, size is good to ~10 %
down to ~1.5 px; fainter blobs measure too big, i.e. range too short.)
"""
from dataclasses import dataclass
import math


@dataclass
class RangeConfig:
    target_size_m: float = 0.35      # drone size the camera sees (m)
    size_uncert_frac: float = 0.25   # 1-sigma uncertainty of that size (0.25 = 25 %)
    psf_px: float = 2.5              # measured size of a POINT hot spot (blur), px
    psf_sigma_px: float = 0.3        # uncertainty of psf_px
    size_sigma_px: float = 0.3       # 1-frame noise of the size measurement, px
    min_resolved_px: float = 1.5     # below this true size (px), only a lower bound is given
    min_snr: float = 30.0            # below this local SNR (det.size_snr), only a lower bound
    smooth_tau_s: float = 0.5        # averaging time for the size (s); 0 = no smoothing
    floor_sigma_px: float = 0.15     # smoothing can't push pixel error below this


class RangeEstimator:
    def __init__(self, cfg: RangeConfig = RangeConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.track_id = None
        self.size_s = None       # smoothed true size (px)
        self.sigma_s = None      # its 1-sigma (px)
        self.snr_s = 0.0         # smoothed SNR
        self.t = None
        self.last = self._empty()

    @staticmethod
    def _empty():
        return {"range_m": None, "range_sigma_m": None, "range_min_m": None,
                "size_px": None, "size_true_px": None, "ang_size_mrad": None,
                "range_status": "none"}

    def update(self, t, det, cam, u, v, track_id):
        """t: frame time (s). det: this frame's Detection for the tracked target
        (None if missed). (u, v): track position. Returns a dict of results."""
        c = self.cfg
        if track_id != self.track_id:          # new target -> start over
            self.reset()
            self.track_id = track_id
        if det is None or det.size_px <= 0:
            if self.last["range_status"] != "none":
                self.last = dict(self.last, range_status="held")   # coasting: keep last
            return self.last

        rad = cam.rad_per_px(u, v)             # angle of one pixel here (fisheye aware)
        L, p = det.size_px, c.psf_px
        true_px = math.sqrt(max(L * L - p * p, 0.0))          # blur removed
        sigma_inst = (math.sqrt((L * c.size_sigma_px) ** 2 + (p * c.psf_sigma_px) ** 2)
                      / max(true_px, c.min_resolved_px))

        # exponential smoothing of the size and SNR over ~smooth_tau_s
        if self.size_s is None or c.smooth_tau_s <= 0:
            a = 1.0
        else:
            a = 1.0 - math.exp(-max(0.0, t - self.t) / c.smooth_tau_s)
        self.size_s = true_px if self.size_s is None else self.size_s + a * (true_px - self.size_s)
        self.snr_s = det.size_snr if a == 1.0 else self.snr_s + a * (det.size_snr - self.snr_s)
        self.t = t
        # averaging shrinks random error by ~sqrt(N frames); a/(2-a) is that factor for
        # exponential smoothing. floor_sigma_px keeps it from claiming too much.
        sigma_s = max(sigma_inst * math.sqrt(a / (2.0 - a)), c.floor_sigma_px) if a < 1 else sigma_inst
        self.sigma_s = sigma_s

        res = self._empty()
        res.update(size_px=round(L, 2), size_true_px=round(self.size_s, 2))

        def rng_from(px):
            return c.target_size_m / (2 * math.tan(px * rad / 2))

        if self.snr_s < c.min_snr or self.size_s < c.min_resolved_px:
            # Too faint or too small to size well. A faint blob measures TOO BIG,
            # which makes range TOO SHORT, so the number is still a safe lower
            # bound: "the drone is at least this far away".
            res.update(range_min_m=round(rng_from(max(self.size_s, c.min_resolved_px)), 1),
                       range_status="low_snr" if self.snr_s < c.min_snr else "unresolved")
            self.last = res
            return res

        alpha = self.size_s * rad                       # angular size (rad)
        rng = rng_from(self.size_s)
        rel = math.sqrt((sigma_s / self.size_s) ** 2 + c.size_uncert_frac ** 2)
        res.update(range_m=round(rng, 2), range_sigma_m=round(rng * rel, 2),
                   ang_size_mrad=round(alpha * 1000, 3), range_status="ok")
        self.last = res
        return res

    def effective_size_at(self, known_range_m):
        """Calibration helper: with the drone at a KNOWN range, this is the
        size (m) to put in DRONE_SIZE_M so the range comes out right."""
        a = self.last.get("ang_size_mrad")
        if a is None:
            return None
        return 2 * known_range_m * math.tan(a / 1000 / 2)
