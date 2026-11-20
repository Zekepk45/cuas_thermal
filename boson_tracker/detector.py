"""
Hot-spot detector for LWIR video ("find the hot blob, compute its center").

A small drone against sky is a few warm pixels on a fairly smooth background.
For each frame the detector:
  1. SPATIAL filter - keeps only warm features smaller than the target window
     (white top-hat). Removes sky gradient, AGC drift, big warm areas.
  2. TEMPORAL filter (camera not moving) - subtracts a slow running average
     of step 1, so anything that sits still (roof corners, tree tops, a
     warm lamp post) fades out and only things that move stay bright.
     The pixels around the current track are frozen in that average, so a
     hovering drone is not absorbed while it is being tracked.
  3. Threshold from the image's own noise (median + k * MAD).
  4. Blob grouping and intensity-weighted (sub-pixel) centroid per blob.

Modes:
  local  (default) - steps 1-4. Small / distant targets, cluttered scenes.
  global           - step 1 replaced by "minus whole-image median". For big,
                     close targets on the bench (hand, mug, drone at 2-10 m).
Set temporal_tau_s = 0 when the camera is on a moving gimbal.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Detection:
    u: float            # sub-pixel centroid, x (pixels)
    v: float            # sub-pixel centroid, y (pixels)
    area: int           # pixels in the blob
    peak: float         # peak contrast above background (gray levels)
    snr: float          # peak / noise sigma
    bbox: tuple         # (x, y, w, h)
    size_px: float = 0.0    # apparent size along the long axis, 4*sqrt(variance) (px, blur included)
    minor_px: float = 0.0   # apparent length of its short axis (pixels)
    size_snr: float = 0.0   # peak / local noise around the blob (used to trust the size)


@dataclass
class DetectorConfig:
    mode: str = "local"           # "local", "global" or "hottest"
    hot_delta: float = 12.0       # hottest mode: gray levels below the frame max that still count
    k_sigma: float = 5.0          # threshold in noise sigmas
    min_area: int = 1
    max_area: int = 5000
    bg_kernel: int = 15           # local mode: largest target size kept (odd, pixels)
    temporal_tau_s: float = 4.0   # static-clutter memory (s); 0 = off (moving camera)
    black_hot: bool = False       # set if the stream uses the black-hot palette
    ignore_border: int = 4        # ignore N pixels at each edge
    max_detections: int = 10
    min_contrast: float = 5.0     # gray levels; floor on the threshold


def measure_size(signal, u, v, bbox, noise=None, pad=3, iters=12):
    """Apparent size of a blob using ADAPTIVE (Gaussian-weighted) moments,
    the method astronomers use to size faint, blurry stars.

    signal: image where the target is brighter than its surroundings.
    Returns (long_axis_px, short_axis_px, local_snr).

    How:
      1. Cut a window around the blob; the median of its outer ring is the
         local background, which is subtracted.
      2. Weight every pixel by a bell curve (Gaussian) centered on the blob,
         then compute the blob's spread (covariance matrix C).
      3. Make the bell curve the same size as the blob (2*C) and repeat until
         it settles. For a Gaussian blob this converges to C = true/2, so the
         true spread is 2*C.
      The bell-curve weight ignores noise far from the blob, and there is no
      brightness threshold, so faint and bright blobs are sized the same way.

    "Size" = 4 * sqrt(variance along the axis), which equals the diameter of a
    solid disc. When an image is blurred, variances ADD, so blur can be removed
    with true = sqrt(seen^2 - blur^2) (range_estimator.py does this).
    """
    x, y, w, h = bbox
    H, W = signal.shape[:2]
    r = int(max(w, h) // 2 + pad + 3)                 # window half-size (px)
    cu, cv_ = int(round(u)), int(round(v))
    x0, y0, x1, y1 = cu - r, cv_ - r, cu + r + 1, cv_ + r + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return 0.0, 0.0, 0.0                          # too close to the edge to measure
    win = signal[y0:y1, x0:x1].astype(np.float64)
    ring = np.concatenate([win[0, :], win[-1, :], win[1:-1, 0], win[1:-1, -1]])
    bg = np.median(ring)
    img = win - bg                                    # local background removed
    ring_noise = max(1.4826 * float(np.median(np.abs(ring - bg))), 0.5)
    local_snr = float(img.max()) / ring_noise
    ys, xs = np.mgrid[0:win.shape[0], 0:win.shape[1]].astype(np.float64)
    mx, my = u - x0, v - y0
    s0 = max(1.0, max(w, h) / 2.355)                  # starting guess for the spread
    M = np.diag([s0 * s0, s0 * s0])                   # weight-function covariance
    C = None
    for _ in range(iters):
        Mi = np.linalg.inv(M)
        dx, dy = xs - mx, ys - my
        q = Mi[0, 0] * dx * dx + 2 * Mi[0, 1] * dx * dy + Mi[1, 1] * dy * dy
        wts = img * np.exp(-0.5 * q)
        tot = wts.sum()
        if tot <= 0:
            return 0.0, 0.0, 0.0
        mx, my = (xs * wts).sum() / tot, (ys * wts).sum() / tot
        dx, dy = xs - mx, ys - my
        C = np.array([[(dx * dx * wts).sum(), (dx * dy * wts).sum()],
                      [(dx * dy * wts).sum(), (dy * dy * wts).sum()]]) / tot
        # next weight = 2*C, but never narrower than 1 px (sigma), so tiny
        # blobs don't collapse onto a single pixel
        ev, evec = np.linalg.eigh(2.0 * C)
        newM = evec @ np.diag(np.maximum(ev, 1.0)) @ evec.T
        if np.allclose(newM, M, rtol=1e-3, atol=1e-4):
            M = newM
            break
        M = newM
    # undo the weighting: for a Gaussian blob, true = inverse(inv(C) - inv(M))
    # (equals 2*C when the weight matches the blob)
    A = np.linalg.inv(C) - np.linalg.inv(M)
    ev, evec = np.linalg.eigh(A)
    if not np.all(np.isfinite(ev)):
        return 0.0, 0.0, 0.0
    lam = 1.0 / np.maximum(ev, 1e-6)                  # variances along the two axes
    lam = np.minimum(lam, 4.0 * float(np.max(np.linalg.eigvalsh(C))))  # guard
    lam = np.sort(np.maximum(lam, 0.0))               # small, large
    return float(4 * np.sqrt(lam[1])), float(4 * np.sqrt(lam[0])), local_snr


class HotSpotDetector:
    def __init__(self, cfg: DetectorConfig = DetectorConfig()):
        self.cfg = cfg
        k = cfg.bg_kernel if cfg.bg_kernel % 2 == 1 else cfg.bg_kernel + 1
        self._se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k),) * 2)
        self._static = None      # running average of the spatial residual
        self._var = None         # running per-pixel noise variance
        self._t_last = None
        self.last_threshold = 0.0
        self.last_noise = 0.0

    def reset_background(self):
        self._static = None

    def detect(self, gray: np.ndarray, t: float = None, protect=None):
        """gray: HxW image.  t: frame time (s), needed for the temporal filter.
        protect: (u, v, radius_px) region kept out of the static average
        (the tracker passes its current target here)."""
        c = self.cfg
        img = gray if gray.dtype == np.uint8 else cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if c.black_hot:
            img = 255 - img
        if c.mode == "hottest":
            return self._detect_hottest(img)

        # 1. spatial
        if c.mode == "global":
            resid = img.astype(np.float32) - float(np.median(img[::4, ::4]))
        else:
            resid = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self._se).astype(np.float32)

        # 2. temporal (static clutter suppression) + per-pixel noise map.
        # Pixels that flicker (video-compression artifacts on sharp edges,
        # swaying branches) learn a higher noise level and need more to trigger.
        sigma = None
        if c.temporal_tau_s > 0 and t is not None:
            if self._static is None:
                self._static = resid.copy()
                self._var = np.full_like(resid, 9.0)
                self._t_last = t
            dt = max(0.0, min(t - self._t_last, 1.0))
            self._t_last = t
            moving = resid - self._static
            if dt > 0:
                a = 1.0 - np.exp(-dt / c.temporal_tau_s)
                av = 1.0 - np.exp(-dt / 1.0)
                new_static = self._static + a * (resid - self._static)
                new_var = self._var + av * (np.minimum(moving * moving, 400.0) - self._var)
                if protect is not None:
                    pu, pv, pr = protect
                    x0, x1 = int(max(0, pu - pr)), int(min(img.shape[1], pu + pr + 1))
                    y0, y1 = int(max(0, pv - pr)), int(min(img.shape[0], pv + pr + 1))
                    new_static[y0:y1, x0:x1] = self._static[y0:y1, x0:x1]
                    new_var[y0:y1, x0:x1] = self._var[y0:y1, x0:x1]
                self._static, self._var = new_static, new_var
            resid = moving
            sigma = np.sqrt(self._var + 1.0)

        # 3. threshold
        if sigma is not None:
            score = resid / sigma
            med = 0.0
            noise = float(np.median(sigma[::3, ::3]))
            mask = ((score > c.k_sigma) & (resid > c.min_contrast)).astype(np.uint8)
            thr = c.k_sigma * noise
        else:
            sample = resid[::3, ::3]
            med = float(np.median(sample))
            noise = max(float(np.median(np.abs(sample - med))) * 1.4826, 0.5)
            score = (resid - med) / noise
            thr = med + max(c.k_sigma * noise, c.min_contrast)
            mask = (resid > thr).astype(np.uint8)
        self.last_threshold, self.last_noise = thr, noise

        b = c.ignore_border
        if b > 0:
            mask[:b, :] = 0; mask[-b:, :] = 0; mask[:, :b] = 0; mask[:, -b:] = 0
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))  # join broken blobs

        # 4. blobs + centroids
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < c.min_area or area > c.max_area:
                continue
            roi = np.where(labels[y:y + h, x:x + w] == i,
                           np.clip(resid[y:y + h, x:x + w] - med, 0, None), 0)
            s = float(roi.sum())
            if s <= 0:
                continue
            ys, xs = np.mgrid[y:y + h, x:x + w]
            peak = float(roi.max())
            snr = float(score[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i].max())
            dets.append(Detection(float((xs * roi).sum() / s), float((ys * roi).sum() / s),
                                  int(area), peak, snr,
                                  (int(x), int(y), int(w), int(h))))
        dets.sort(key=lambda d: d.snr, reverse=True)
        dets = dets[:c.max_detections]
        sig = resid - med
        for d in dets:                       # size for range-from-size
            d.size_px, d.minor_px, d.size_snr = measure_size(sig, d.u, d.v, d.bbox, noise)
        return dets

    # ------------------------------------------------------------------ hottest
    def _detect_hottest(self, img):
        """Return ONLY the single hottest object in the frame (e.g. a hotplate).
        No clutter filtering and no fading of still objects: the brightest
        blob wins every frame, whatever its size."""
        c = self.cfg
        sm = cv2.GaussianBlur(img, (5, 5), 0)          # ignore single noisy pixels
        b = c.ignore_border
        core = sm[b:-b, b:-b] if b > 0 else sm
        maxv = float(core.max())
        med = float(np.median(sm[::4, ::4]))
        noise = max(float(np.median(np.abs(sm[::4, ::4] - med))) * 1.4826, 0.5)
        self.last_noise = noise
        if maxv - med < max(c.min_contrast, c.k_sigma * noise):
            self.last_threshold = maxv
            return []                                   # nothing stands out
        thr = maxv - c.hot_delta
        self.last_threshold = thr
        mask = (sm >= thr).astype(np.uint8)
        if b > 0:
            mask[:b, :] = 0; mask[-b:, :] = 0; mask[:, :b] = 0; mask[:, -b:] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < c.min_area:
                continue
            sub = sm[y:y + h, x:x + w].astype(np.float32)
            lab = labels[y:y + h, x:x + w] == i
            wts = np.where(lab, sub - thr + 1.0, 0.0)
            score = float(wts.sum())                    # hot AND big beats a hot speck
            if best is None or score > best[0]:
                ys, xs = np.mgrid[y:y + h, x:x + w]
                s = float(wts.sum())
                peak = float(sub[lab].max())
                best = (score, Detection(float((xs * wts).sum() / s), float((ys * wts).sum() / s),
                                         int(area), peak - med, (peak - med) / noise,
                                         (int(x), int(y), int(w), int(h))))
        if not best:
            return []
        d = best[1]
        d.size_px, d.minor_px, d.size_snr = measure_size(img.astype(np.float32) - med, d.u, d.v, d.bbox, noise)
        return [d]
