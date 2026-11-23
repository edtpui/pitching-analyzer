"""Time-series utilities: gap filling, zero-phase low-pass filtering, resampling, derivatives."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Cut-off frequencies used for BOTH the video pipeline and the simulated Driveline reference, so the
# reference distributions carry the same smoothing as a user's clip.
CUTOFF_BODY_HZ = 12.0
CUTOFF_ARM_HZ = 18.0


def fill_gaps(x: np.ndarray, t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than `max_gap_s` along axis 0; longer runs stay NaN.
    Leading/trailing NaNs are left untouched. Works on (T,) or (T, ...) arrays."""
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(len(t), -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = ~np.isfinite(col)
        if not bad.any() or bad.all():
            continue
        good = np.flatnonzero(~bad)
        interp = np.interp(t, t[good], col[good])
        # walk NaN runs
        i = 0
        n = len(col)
        while i < n:
            if not bad[i]:
                i += 1
                continue
            j = i
            while j < n and bad[j]:
                j += 1
            interior = i > 0 and j < n
            if interior and (t[j] - t[i - 1]) <= max_gap_s:
                col[i:j] = interp[i:j]
            i = j
        flat[:, c] = col
    return flat.reshape(x.shape)


MIN_FILTER_SAMPLES = 24


def max_gap_s(fps: float) -> float:
    """Longest run of missing samples that is bridged by interpolation (video and simulation alike)."""
    return max(0.1, 3.0 / fps)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass along axis 0 (a 2nd-order design run forward and
    backward). NaN segments are bridged by interpolation for filtering and restored afterwards. The
    cut-off is clamped to 0.8 x Nyquist for low frame rates."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < MIN_FILTER_SAMPLES:
        return x.copy()
    nyq = fs / 2.0
    wn = min(cutoff_hz, 0.8 * nyq) / nyq
    sos = butter(2, wn, btype="low", output="sos")
    flat = x.reshape(n, -1).copy()
    out = np.full_like(flat, np.nan)
    idx = np.arange(n)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = np.isfinite(col)
        if ok.sum() < MIN_FILTER_SAMPLES:
            out[:, c] = col
            continue
        filled = np.interp(idx, idx[ok], col[ok])
        y = sosfiltfilt(sos, filled)
        y[~ok] = np.nan
        out[:, c] = y
    return out.reshape(x.shape)


def smooth_skeleton(pos: np.ndarray, fs: float) -> np.ndarray:
    """Low-pass a (T, 12, 3) canonical skeleton: arm joints at CUTOFF_ARM_HZ, the rest at CUTOFF_BODY_HZ."""
    from .skeleton import ARM_JOINTS, J

    out = lowpass(pos, fs, CUTOFF_BODY_HZ)
    arm = [J[n] for n in ARM_JOINTS]
    out[:, arm] = lowpass(pos[:, arm], fs, CUTOFF_ARM_HZ)
    return out


def resample(x: np.ndarray, t: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linear resampling along axis 0 onto `t_new` (NaN-aware: output NaN where neighbours are NaN)."""
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(t), -1)
    out = np.empty((len(t_new), flat.shape[1]))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            out[:, c] = np.nan
            continue
        out[:, c] = np.interp(t_new, t[ok], col[ok])
        # invalidate samples whose nearest source sample was missing
        nearest = np.clip(np.searchsorted(t, t_new), 0, len(t) - 1)
        out[~ok[nearest], c] = np.nan
    return out.reshape((len(t_new),) + x.shape[1:])


def derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """First derivative along axis 0 using second-order central differences (non-uniform t allowed)."""
    x = np.asarray(x, dtype=float)
    if len(t) < 3:
        return np.full_like(x, np.nan)
    return np.gradient(x, t, axis=0)


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    """np.unwrap for degree series containing NaNs (NaNs kept in place)."""
    a = np.asarray(a, dtype=float)
    out = a.copy()
    ok = np.isfinite(a)
    if ok.sum() > 1:
        out[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
    return out
