"""Session-level scoring. Pure numpy so it can be tested without torch/YOLO.

The formula is copied from run_live_analysis.py:
    WCI = 0.50 * mean + 0.30 * best 1-second window mean + 0.20 * peak
    fake if WCI > 0.45 and (spike_ratio > 0.12 or peak > 0.85)
"""
import numpy as np

SPIKE_THRESHOLD = 0.50


def aggregate(scores, sample_fps, min_faces=5):
    n = len(scores)
    base = {
        "faces_evaluated": n,
        "wci": 0.0,
        "avg": 0.0,
        "max_burst": 0.0,
        "peak": 0.0,
        "spike_ratio": 0.0,
    }

    if n < min_faces:
        base.update(
            verdict="inconclusive",
            reason=(
                f"A usable face was found in only {n} sampled frame(s); "
                f"at least {min_faces} are needed. Try a clip where the person's "
                "face is larger, sharper and on screen for longer."
            ),
        )
        return base

    arr = np.asarray(scores, dtype=np.float64)
    avg = float(arr.mean())
    peak = float(arr.max())
    spike_ratio = float((arr > SPIKE_THRESHOLD).mean())

    window = max(1, int(round(sample_fps * 1.0)))
    if n >= window:
        max_burst = float(np.convolve(arr, np.ones(window) / window, mode="valid").max())
    else:
        max_burst = avg

    wci = 0.50 * avg + 0.30 * max_burst + 0.20 * peak
    is_fake = wci > 0.45 and (spike_ratio > 0.12 or peak > 0.85)

    base.update(
        wci=wci,
        avg=avg,
        max_burst=max_burst,
        peak=peak,
        spike_ratio=spike_ratio,
        verdict="synthetic" if is_fake else "authentic",
        reason=None,
    )
    return base
