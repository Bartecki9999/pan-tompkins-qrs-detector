"""Plotting helpers for the detector output and the filter responses."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

from pan_tompkins import design_filters


def plot_detection(ecg, detection, reference, fs, title, n_samples=2000):
    """Plot the raw ECG with detections, then each processing stage below it.

    Only the first `n_samples` samples are shown (about 5.5 s at 360 Hz),
    which is enough to see individual beats.
    """
    t = np.arange(n_samples) / fs

    r_in_view = detection.r_peaks[detection.r_peaks < n_samples]
    ref_in_view = reference[reference < n_samples]

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(t, ecg[:n_samples], label="raw ECG", linewidth=1)
    ax.plot(r_in_view / fs, ecg[r_in_view], "ro", label="detected R peaks")
    ax.plot(ref_in_view / fs, ecg[ref_in_view], "gx", markersize=10, label="reference")
    ax.legend(loc="upper right")
    ax.set_ylabel("ECG [mV]")

    stages = [
        (detection.filtered, "band-pass\n5-15 Hz"),
        (detection.derivative, "derivative"),
        (detection.squared, "squared"),
        (detection.integrated, "integrated"),
    ]
    for ax, (signal, label) in zip(axes[1:], stages):
        ax.plot(t, signal[:n_samples], linewidth=1)
        ax.set_ylabel(label)

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    return fig


def plot_filter_response(fs):
    """Magnitude and phase response of the low-pass and high-pass filters."""
    h_lp, h_hp = design_filters(fs)
    f, H_lp = freqz(h_lp, worN=2048, fs=fs)
    _, H_hp = freqz(h_hp, worN=2048, fs=fs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    for col, (H, name) in enumerate([(H_lp, "Low-pass (15 Hz)"), (H_hp, "High-pass (5 Hz)")]):
        magnitude, phase = axes[0, col], axes[1, col]

        magnitude.plot(f, 20 * np.log10(np.abs(H) + 1e-12))
        magnitude.set_title(name)
        magnitude.set_ylim(-80, 5)
        magnitude.grid(True)

        phase.plot(f, np.unwrap(np.angle(H)))
        phase.set_xlabel("frequency [Hz]")
        phase.grid(True)

    axes[0, 0].set_ylabel("magnitude [dB]")
    axes[1, 0].set_ylabel("phase [rad]")

    fig.tight_layout()
    return fig
