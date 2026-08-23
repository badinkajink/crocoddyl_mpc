#!/usr/bin/env python3
"""Spectral arc length (SPARC), the smoothness metric the hardware runs report.

Balasubramanian, Melendez-Calderon, Roby-Brami & Burdet, "On the analysis of
movement smoothness", J. NeuroEng. Rehabil. 12:112 (2015). This is their
reference algorithm, kept faithful so a sim number and a hardware number mean
the same thing:

  * take the speed profile of the movement,
  * FFT it with `padlevel` zero-padding octaves,
  * normalise the magnitude spectrum by its DC value,
  * cut at the LAST frequency below `fc` whose normalised magnitude still
    exceeds `amp_th` (the adaptive cutoff -- a fixed cutoff makes the metric
    depend on how much empty spectrum you happened to include),
  * return minus the arc length of the normalised spectrum over that band.

SPARC is NEGATIVE and DIMENSIONLESS. Less negative = smoother; a clean
minimum-jerk reach is about -1.5, and it grows in magnitude with every extra
sub-movement. It is amplitude- and duration-invariant by construction, which is
why it survives the sim-to-real comparison that a jitter-in-millimetres number
does not.
"""
import numpy as np


def sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """(sparc, (f, Mf), (f_sel, Mf_sel)) for a speed profile sampled at fs Hz."""
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 4 or not np.any(speed > 0):
        return float("nan"), (None, None), (None, None)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    if Mf[0] <= 0:
        return float("nan"), (None, None), (None, None)
    Mf = Mf / np.max(Mf)

    inx = np.where(f <= fc)[0]
    f_sel, Mf_sel = f[inx], Mf[inx]
    above = np.where(Mf_sel >= amp_th)[0]
    if len(above) < 2:
        return float("nan"), (f, Mf), (None, None)
    f_sel = f_sel[above[0]:above[-1] + 1]
    Mf_sel = Mf_sel[above[0]:above[-1] + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 0:
        return float("nan"), (f, Mf), (f_sel, Mf_sel)
    val = -np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(Mf_sel) ** 2))
    return float(val), (f, Mf), (f_sel, Mf_sel)


def movement_window(speed, frac=0.05):
    """Onset/offset indices of the movement: the span where speed exceeds
    `frac` of its peak.

    SPARC is defined on a movement, not on a recording. These rollouts reach in
    ~3 s and then hold for ~17 s; run over the whole trace the metric is
    dominated by the stationary tail and stops describing the reach."""
    speed = np.asarray(speed, dtype=float)
    pk = speed.max()
    if pk <= 0:
        return 0, len(speed) - 1
    on = np.where(speed >= frac * pk)[0]
    return int(on[0]), int(on[-1])
