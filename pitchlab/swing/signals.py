"""The app's time-series utilities, with skeleton smoothing on the swing's own joint set."""
from __future__ import annotations

import numpy as np

from ..signals import (CUTOFF_ARM_HZ, CUTOFF_BODY_HZ, MIN_FILTER_SAMPLES, derivative, fill_gaps, lowpass, max_gap_s,
                       resample, unwrap_deg)

__all__ = ["CUTOFF_ARM_HZ", "CUTOFF_BODY_HZ", "MIN_FILTER_SAMPLES", "derivative", "fill_gaps", "lowpass", "max_gap_s",
           "resample", "smooth_skeleton", "unwrap_deg"]


def smooth_skeleton(pos: np.ndarray, fs: float) -> np.ndarray:
    """Low-pass a (T, 12, 3) canonical skeleton: arm joints at CUTOFF_ARM_HZ, the rest at CUTOFF_BODY_HZ."""
    from .skeleton import ARM_JOINTS, J

    out = lowpass(pos, fs, CUTOFF_BODY_HZ)
    arm = [J[n] for n in ARM_JOINTS]
    out[:, arm] = lowpass(pos[:, arm], fs, CUTOFF_ARM_HZ)
    return out
