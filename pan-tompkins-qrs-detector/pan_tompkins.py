"""
A small implementation of the Pan-Tompkins QRS detector.

Reference:
    J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm",
    IEEE Transactions on Biomedical Engineering, 32(3), 1985, pp. 230-236.

The pipeline is the classic one:
    band-pass (5-15 Hz) -> derivative -> squaring -> moving-window integration
    -> adaptive thresholds with search-back.

A few things differ from the paper on purpose. The paper uses cheap integer
filters designed for 200 Hz hardware; here the band-pass is built from two
windowed-sinc FIR filters so it works at any sampling rate (the MIT-BIH
records used here are sampled at 360 Hz).
"""

from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks


# --------------------------------------------------------------------------
# Loading data
# --------------------------------------------------------------------------

def load_signal(path):
    """Read a one-column text file with one ECG sample (in mV) per line."""
    return np.loadtxt(path, dtype=float)


def load_annotations(path):
    """Read reference beat positions (sample indices) from an annotation file.

    Each line looks like "<sample index><whitespace><beat symbol>"; only the
    sample index is used.
    """
    with open(path, "r") as f:
        return np.array([int(line.split()[0]) for line in f if line.strip()])


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

def design_filters(fs, half_order=50, low_cut=15.0, high_cut=5.0):
    """Build the two FIR filters that together make a 5-15 Hz band-pass.

    Both are windowed-sinc filters with a Hamming window and 2 * half_order + 1
    taps. They are symmetric, so they have linear phase and a constant delay of
    `half_order` samples.

    Returns
    -------
    h_lp : low-pass impulse response (cutoff `low_cut` Hz)
    h_hp : high-pass impulse response (cutoff `high_cut` Hz)
    """
    k = np.arange(-half_order, half_order + 1)
    window = np.hamming(len(k))

    # Ideal low-pass: sin(2*pi*fc*k) / (pi*k). np.sinc handles the k = 0 case.
    fc_low = low_cut / fs
    h_lp = 2 * fc_low * np.sinc(2 * fc_low * k)

    # High-pass by spectral inversion: a unit impulse minus a low-pass.
    fc_high = high_cut / fs
    h_hp = -2 * fc_high * np.sinc(2 * fc_high * k)
    h_hp[half_order] += 1.0

    return h_lp * window, h_hp * window


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

class Detection(NamedTuple):
    """Everything the detector produces, including the intermediate signals."""
    r_peaks: np.ndarray    # sample indices of detected R peaks
    filtered: np.ndarray   # after the 5-15 Hz band-pass
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray  # moving-window integration output


def detect_r_peaks(ecg, fs, integration_ms=150):
    """Find R peaks in a single-lead ECG.

    Parameters
    ----------
    ecg : 1-D array, the raw signal
    fs : sampling frequency in Hz
    integration_ms : length of the moving-window integrator in milliseconds
        (the paper suggests about 150 ms)

    Returns
    -------
    Detection
    """
    ecg = np.asarray(ecg, dtype=float)

    # 1. Band-pass 5-15 Hz.
    # The filters are symmetric and mode="same" re-centres the output, so the
    # filtered signal is not shifted in time relative to the input.
    h_lp, h_hp = design_filters(fs)
    filtered = np.convolve(ecg, h_lp, mode="same")
    filtered = np.convolve(filtered, h_hp, mode="same")

    # 2. Five-point derivative (eq. 5 in the paper):
    #    y[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / 8
    derivative = np.zeros_like(filtered)
    derivative[2:-2] = (-filtered[:-4] - 2 * filtered[1:-3]
                        + 2 * filtered[3:-1] + filtered[4:]) / 8

    # 3. Squaring (eq. 6): makes everything positive and stresses steep slopes.
    squared = derivative ** 2

    # 4. Moving-window integration (eq. 7).
    window = int(integration_ms / 1000 * fs)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    # 5. Candidate peaks: local maxima at least 200 ms apart. This is also the
    # refractory period; the heart can't produce two QRS complexes closer.
    refractory = int(0.2 * fs)
    candidates, _ = find_peaks(integrated, distance=refractory)

    r_peaks = _threshold_candidates(integrated, candidates, fs, refractory)
    r_peaks = _refine_peak_positions(ecg, r_peaks, fs)

    return Detection(r_peaks, filtered, derivative, squared, integrated)


def _threshold_candidates(integrated, candidates, fs, refractory):
    """Decide which candidate peaks are QRS complexes (adaptive thresholds).

    Names follow the paper: SPKI / NPKI are running estimates of the signal and
    noise peak levels, threshold_1 (I1 in the paper) is the main threshold and
    threshold_2 (I2) is the lower one used during search-back.
    """
    # Start the estimates from the first two seconds.
    start = integrated[:2 * fs]
    spki = start.max()
    npki = start.mean()
    threshold_1 = npki + 0.25 * (spki - npki)   # eq. 10
    threshold_2 = 0.5 * threshold_1             # eq. 11

    accepted = []
    noise_peaks = []   # rejected candidates since the last accepted beat

    for p in candidates:
        height = integrated[p]

        if height > threshold_1:
            spki = 0.125 * height + 0.875 * spki   # eq. 8

            # Search-back: if the gap since the last beat is much longer than
            # the recent RR average, we probably skipped a beat. Look through
            # the rejected peaks for the biggest one above the lower threshold.
            if len(accepted) >= 2:
                rr_mean = np.mean(np.diff(accepted[-9:]))   # last 8 intervals
                if p - accepted[-1] > 1.66 * rr_mean:
                    best = None
                    for q in noise_peaks:
                        in_range = accepted[-1] + refractory < q < p - refractory
                        if in_range and integrated[q] > threshold_2:
                            if best is None or integrated[q] > integrated[best]:
                                best = q
                    if best is not None:
                        accepted.append(best)
                        spki = 0.25 * integrated[best] + 0.75 * spki

            accepted.append(p)
            noise_peaks = []
        else:
            npki = 0.125 * height + 0.875 * npki   # eq. 9
            noise_peaks.append(p)

        threshold_1 = npki + 0.25 * (spki - npki)
        threshold_2 = 0.5 * threshold_1

    return np.array(accepted, dtype=int)


def _refine_peak_positions(ecg, peaks, fs):
    """Move each detection onto the actual R peak in the raw ECG.

    The integrated signal peaks somewhere in the middle of the QRS complex, so
    the raw signal is searched +-75 ms around it for the sample that is
    furthest from the local baseline (taken as the median). The absolute value
    is used so that negative R waves (inverted leads, PVCs) are caught too.
    """
    half_window = int(0.075 * fs)
    half_baseline = int(0.2 * fs)

    refined = []
    for p in peaks:
        baseline = np.median(ecg[max(p - half_baseline, 0):p + half_baseline])
        a = max(p - half_window, 0)
        b = min(p + half_window, len(ecg))
        refined.append(a + np.argmax(np.abs(ecg[a:b] - baseline)))

    return np.array(refined, dtype=int)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

class Metrics(NamedTuple):
    mae_ms: float   # mean absolute timing error of matched beats
    std_ms: float   # standard deviation of the (signed) timing error
    tp: int         # true positives
    fp: int         # false positives
    fn: int         # false negatives

    @property
    def sensitivity(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0


def evaluate(detected, reference, fs, tolerance_ms=75):
    """Compare detections with reference annotations.

    Every reference beat is matched to the nearest detection. It counts as a
    hit if that detection is within `tolerance_ms` and has not been used for
    another reference beat already.
    """
    detected = np.asarray(detected)
    reference = np.asarray(reference)
    tolerance = int(tolerance_ms / 1000 * fs)

    errors = []
    used = set()

    if len(detected):
        for ref in reference:
            i = int(np.argmin(np.abs(detected - ref)))
            if abs(detected[i] - ref) <= tolerance and i not in used:
                used.add(i)
                errors.append(detected[i] - ref)

    errors_ms = np.array(errors, dtype=float) / fs * 1000
    tp = len(errors_ms)
    fp = len(detected) - tp
    fn = len(reference) - tp

    if tp == 0:
        return Metrics(float("nan"), float("nan"), tp, fp, fn)
    return Metrics(float(np.mean(np.abs(errors_ms))), float(np.std(errors_ms)), tp, fp, fn)
