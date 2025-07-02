import numpy as np
from obspy import Stream, Trace
from typing import Tuple


# ===============================
# === Waveform Integration ===
# ===============================

def integrate_stream(stream: Stream, waveform_type: str) -> Stream:
    """Integrates the stream depending on waveform type: acc, vel, or dis."""
    stream = stream.copy()
    stream.detrend("demean")

    if waveform_type == "acc":
        return stream
    elif waveform_type == "vel":
        stream.taper(max_percentage=0.05)
        stream.integrate()
        stream.filter("highpass", freq=0.075)
    elif waveform_type == "dis":
        stream.taper(max_percentage=0.05)
        stream.integrate()
        stream.filter("highpass", freq=0.075)
        stream.integrate()
        stream.filter("highpass", freq=0.075)
    return stream

# ===============================
# === FFT Analysis ===
# ===============================

def fft(trace: Trace, n: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Performs FFT and returns amplitude and frequency."""
    y_fft = np.fft.fft(trace.data)
    freq = np.fft.fftfreq(n, d=dt)
    amp = np.abs(y_fft / (n / 2))[1:n // 2]
    return amp, freq[1:n // 2]

def fft_divide(wv1: np.ndarray, wv2: np.ndarray) -> np.ndarray:
    """Safely divides two FFT amplitude arrays."""
    if np.any(wv2 == 0):
        raise ZeroDivisionError("wv2 contains zero values, division not possible.")
    return wv1 / wv2

# ===============================
# === Smoothing Average ===
# ===============================

def smooth_moving_average(data, window_size=5):
    assert window_size % 2 == 1, "Window size 必須為奇數"
    pad = window_size // 2
    padded_data = np.pad(data, pad_width=pad, mode='edge')
    smoothed = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
    return smoothed