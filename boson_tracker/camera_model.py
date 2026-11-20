"""
Camera model for the Boson / Boson+ 640: converts a pixel location into a
line-of-sight (LOS) angle relative to the camera boresight, INCLUDING lens
distortion (the fisheye warping of the wide 4.9 mm lens).

Block A4 (Thermal Processing) -> the angle measurement that A6 (Fusion) consumes.

Frame convention (matches the NED / FRD convention used in last year's code):
  FRD = Forward, Right, Down, attached to the camera.
  az_deg  : + to the RIGHT of boresight
  el_deg  : + ABOVE boresight
  los_frd : unit vector [forward, right, down]

Why distortion matters
----------------------
FLIR rates the 4.9 mm lens at 95 deg horizontal FOV. A plain pinhole model with
f = 4.9 mm and 12 um pixels gives only ~76 deg, so the old code put a target at
the image edge ~10 deg away from where it really was. A fisheye lens squeezes
angles toward the edges, so each pixel near the edge covers MORE angle than a
pixel at the center. This model undoes that.

Two lens models are supported:
  "fisheye" - OpenCV fisheye (Kannala-Brandt):
              theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
              theta   = true angle off boresight, theta_d * f = pixel radius
  "pinhole" - OpenCV standard radial/tangential model (k1, k2, p1, p2, k3).
              Fine for the narrow 50 mm pod lens.

Accuracy: until you run calibrate_lens.py, the 4.9 mm lens uses a one-number
fit (k1) chosen so the full width matches FLIR's 95 deg. Expect roughly
+-1 deg error near the edges. After calibration (--calib lens_cal.json) it
should be well under 0.2 deg (about 1 pixel).
"""
from dataclasses import dataclass, field
import json
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class LensProfile:
    key: str
    description: str
    focal_mm: float
    model: str = "pinhole"            # "fisheye" or "pinhole"
    dist: tuple = (0.0, 0.0, 0.0, 0.0)  # fisheye: k1..k4   pinhole: k1, k2, p1, p2, k3
    hfov_spec_deg: float = None       # manufacturer's HFOV, for the startup check
    pixel_pitch_um: float = 12.0      # Boson / Boson+ 640 pixel pitch
    native_w: int = 640
    native_h: int = 512


def _fisheye_k1_for_hfov(focal_mm, pitch_um, native_w, hfov_deg):
    """One-term fisheye fit: pick k1 so the image edge lands at hfov/2."""
    f_px = focal_mm * 1000.0 / pitch_um
    theta = math.radians(hfov_deg / 2.0)             # true edge angle
    theta_d = ((native_w - 1) / 2.0) / f_px           # distorted edge "angle"
    return (theta_d / theta - 1.0) / theta ** 2


# =============================================================================
#  LENS PARAMETERS  -  edit here if you get better numbers (or use --calib)
# =============================================================================
PROFILES = {
    # The lens on the ModalAI Seeker drone (printed "4.9mm" on the lens ring),
    # FLIR part 20640A095 / 22640AS95: 95 deg HFOV.
    "seeker-4.9mm": LensProfile(
        "seeker-4.9mm", "Boson 640, 4.9 mm 95 deg fisheye lens (Seeker drone)", 4.9,
        model="fisheye",
        dist=(_fisheye_k1_for_hfov(4.9, 12.0, 640, 95.0), 0.0, 0.0, 0.0),
        hfov_spec_deg=95.0),
    # The lens planned for the gimbaled pod cameras. Narrow lens, so distortion
    # is small; calibrate it once the pods are built.
    "pod-50mm": LensProfile(
        "pod-50mm", "Boson+ 640, 50 mm lens (sensor pod)", 50.0,
        model="pinhole", dist=(0.0, 0.0, 0.0, 0.0, 0.0)),
}
# =============================================================================


class CameraModel:
    """Turns pixels into angles, with lens distortion removed.

    calib_path: optional JSON written by calibrate_lens.py. When given, its
    measured focal lengths, center and distortion replace the profile's.
    """

    def __init__(self, profile: LensProfile, frame_w: int, frame_h: int, calib_path=None):
        self.profile = profile
        self.w, self.h = frame_w, frame_h
        self.calibrated = False
        self.model = profile.model
        f_native_px = profile.focal_mm * 1000.0 / profile.pixel_pitch_um
        fx = f_native_px * frame_w / profile.native_w
        fy = f_native_px * frame_h / profile.native_h
        cx, cy = (frame_w - 1) / 2.0, (frame_h - 1) / 2.0
        dist = list(profile.dist)

        if calib_path:
            with open(calib_path) as f:
                cal = json.load(f)
            sx, sy = frame_w / cal["image_w"], frame_h / cal["image_h"]
            K = cal["K"]
            fx, fy = K[0][0] * sx, K[1][1] * sy
            cx, cy = K[0][2] * sx, K[1][2] * sy
            dist = list(cal["D"])
            self.model = cal["model"]
            self.calibrated = True
            self.calib_rms_px = cal.get("rms_px")

        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.D = np.array(dist, dtype=np.float64).reshape(-1, 1)

    # --- geometry -----------------------------------------------------------
    def undistort(self, u: float, v: float):
        """Pixel (u, v) -> ideal (distortion-free) normalized coordinates
        x = tan(angle right), y = tan(angle down)."""
        pts = np.array([[[u, v]]], dtype=np.float64)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D)
        else:
            out = cv2.undistortPoints(pts, self.K, self.D)
        x, y = out[0, 0]
        return float(x), float(y)

    def pixel_to_los(self, u: float, v: float):
        """Return (az_deg, el_deg, los_frd) for pixel (u, v)."""
        x, y = self.undistort(u, v)
        n = math.sqrt(1.0 + x * x + y * y)
        los = (1.0 / n, x / n, y / n)
        az = math.degrees(math.atan2(x, 1.0))
        el = math.degrees(math.atan2(-y, math.hypot(1.0, x)))
        return az, el, los

    @staticmethod
    def _angle_between(a, b):
        dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
        return math.acos(dot)

    def rad_per_px(self, u: float, v: float) -> float:
        """How much angle (radians) ONE pixel covers at this spot in the image.
        On a fisheye lens this grows toward the edges."""
        _, _, l0 = self.pixel_to_los(u, v)
        _, _, lu = self.pixel_to_los(u + 0.5, v)
        _, _, lv = self.pixel_to_los(u, v + 0.5)
        return (self._angle_between(l0, lu) + self._angle_between(l0, lv))  # 2 x 0.5 px, averaged

    def pixel_sigma_to_deg(self, sigma_px: float, u: float = None, v: float = None) -> float:
        """Angular 1-sigma for a pixel 1-sigma, at pixel (u, v) (default: center)."""
        if u is None:
            u, v = self.cx, self.cy
        return math.degrees(sigma_px * self.rad_per_px(u, v))

    def angle_rates_dps(self, u, v, du, dv, h=0.05):
        """Pixel velocity (px/s) -> az and el rates (deg/s), distortion included."""
        az0, el0, _ = self.pixel_to_los(u, v)
        az1, el1, _ = self.pixel_to_los(u + du * h, v + dv * h)
        return (az1 - az0) / h, (el1 - el0) / h

    @property
    def ifov_mrad(self) -> float:
        """Angle per pixel at the image center (mrad)."""
        return 1000.0 * self.rad_per_px(self.cx, self.cy)

    @property
    def hfov_deg(self) -> float:
        _, _, a = self.pixel_to_los(0.0, self.cy)
        _, _, b = self.pixel_to_los(self.w - 1.0, self.cy)
        return math.degrees(self._angle_between(a, b))

    @property
    def vfov_deg(self) -> float:
        _, _, a = self.pixel_to_los(self.cx, 0.0)
        _, _, b = self.pixel_to_los(self.cx, self.h - 1.0)
        return math.degrees(self._angle_between(a, b))

    def pixels_on_target(self, size_m: float, range_m: float, u=None, v=None) -> float:
        """How many pixels wide a target of size_m appears at range_m."""
        if u is None:
            u, v = self.cx, self.cy
        return (2 * math.atan(size_m / (2 * range_m))) / self.rad_per_px(u, v)

    def summary(self) -> str:
        src = (f"CALIBRATED (rms {self.calib_rms_px:.2f} px)" if self.calibrated
               else "nominal lens model (run calibrate_lens.py for best accuracy)")
        spec = (f", spec {self.profile.hfov_spec_deg:.0f} deg"
                if self.profile.hfov_spec_deg and not self.calibrated else "")
        return (f"{self.profile.description}: {self.w}x{self.h}, {self.model} model, {src}\n"
                f"  center IFOV={self.ifov_mrad:.3f} mrad/px, "
                f"HFOV={self.hfov_deg:.1f} deg{spec}, VFOV={self.vfov_deg:.1f} deg")
