import numpy as np
from scipy.signal import butter, filtfilt

def compute_resp_waveform(phi, fs=50.0, do_smooth=True, cutoff=1.0, order=2):
    """
    Compute the respiratory waveform by differentiating the breathing angle φ(t).
    Optionally apply a low-pass or band-pass filter to remove high-frequency noise.

    :param phi: 1D NumPy array of breathing angles φₜ (radians).
    :param fs: Sampling frequency in Hz (default 50).
    :param do_smooth: Whether to apply a smoothing filter after differentiation.
    :param cutoff: Cutoff frequency in Hz for the low-pass filter.
    :param order: Order of the Butterworth filter.
    :return: 1D NumPy array of respiratory waveform samples ωₜ.
    """
    # 1) Numerically differentiate φ(t) to get ω(t)
    #    np.gradient(., dx) uses central differences for interior points.
    dt = 1.0 / fs
    raw_derivative = np.gradient(phi, dt)  # shape same as φ

    # 2) (Optional) Low-pass filter the derivative to keep only breathing frequencies
    #    (e.g. < 1 Hz or < 2 Hz). In normal adult breathing, rates are typically < 0.5 Hz (30 BPM).
    if do_smooth:
        nyquist = 0.5 * fs
        norm_cutoff = cutoff / nyquist

        # Butterworth low-pass filter design
        b, a = butter(order, norm_cutoff, btype='low', analog=False)
        resp_waveform = filtfilt(b, a, raw_derivative)
    else:
        resp_waveform = raw_derivative

    return resp_waveform