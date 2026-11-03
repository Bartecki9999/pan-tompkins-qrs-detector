import numpy as np
import pytest

from pan_tompkins import design_filters, detect_r_peaks, evaluate

FS = 360


def synthetic_ecg(n_beats=30, rr_s=0.8, seed=0):
    """A crude fake ECG: narrow spikes on top of a slow baseline wander plus noise."""
    rng = np.random.default_rng(seed)
    n = int((n_beats + 2) * rr_s * FS)
    t = np.arange(n) / FS

    ecg = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.02 * rng.standard_normal(n)

    beat_positions = (np.arange(n_beats) * rr_s * FS + rr_s * FS).astype(int)
    kernel_t = np.arange(-15, 16) / FS
    kernel = np.exp(-(kernel_t / 0.012) ** 2)   # ~25 ms wide bump
    for p in beat_positions:
        ecg[p - 15:p + 16] += kernel
    return ecg, beat_positions


def test_filters_are_symmetric_and_have_expected_length():
    h_lp, h_hp = design_filters(FS, half_order=50)
    assert len(h_lp) == len(h_hp) == 101
    np.testing.assert_allclose(h_lp, h_lp[::-1], atol=1e-12)
    np.testing.assert_allclose(h_hp, h_hp[::-1], atol=1e-12)


def test_lowpass_passes_dc_and_highpass_blocks_it():
    h_lp, h_hp = design_filters(FS)
    assert h_lp.sum() == pytest.approx(1.0, abs=0.05)
    assert abs(h_hp.sum()) < 0.1


def test_detects_every_beat_in_clean_synthetic_signal():
    ecg, beats = synthetic_ecg()
    detection = detect_r_peaks(ecg, FS)
    m = evaluate(detection.r_peaks, beats, FS)
    assert m.fn == 0
    assert m.fp == 0
    assert m.mae_ms < 10


def test_output_signals_have_the_same_length_as_input():
    ecg, _ = synthetic_ecg(n_beats=10)
    d = detect_r_peaks(ecg, FS)
    for signal in (d.filtered, d.derivative, d.squared, d.integrated):
        assert len(signal) == len(ecg)


def test_evaluate_perfect_match():
    ref = np.array([100, 500, 900])
    m = evaluate(ref.copy(), ref, FS)
    assert (m.tp, m.fp, m.fn) == (3, 0, 0)
    assert m.mae_ms == 0.0
    assert m.sensitivity == 1.0 and m.precision == 1.0


def test_evaluate_counts_misses_and_extras():
    ref = np.array([100, 500, 900])
    detected = np.array([102, 900, 2000])   # 500 missed, 2000 is extra
    m = evaluate(detected, ref, FS)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)


def test_evaluate_does_not_match_one_detection_twice():
    ref = np.array([100, 110])
    detected = np.array([105])
    m = evaluate(detected, ref, FS)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)


def test_evaluate_with_no_detections():
    m = evaluate(np.array([]), np.array([100, 200]), FS)
    assert (m.tp, m.fp, m.fn) == (0, 0, 2)
