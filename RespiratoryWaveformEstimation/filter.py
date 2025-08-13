import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=2.0, fs=12.5, order=2):
    """
    Butterworth low-pass filter using filtfilt (zero-phase).
    :param data: 1D NumPy array of samples.
    :param cutoff: Cutoff frequency in Hz.
    :param fs: Sampling frequency in Hz.
    :param order: Filter order.
    :return: Filtered 1D array (same length as input).
    """
    # Normalize the frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # Design filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def filter_accelerometer_data(accel_data, cutoff=2.0, fs=50.0, order=2):
    """
    Apply low-pass filtering to each axis of a tri-axial accelerometer signal.
    :param accel_data: Nx3 NumPy array of raw accelerations.
    :param cutoff: Cutoff frequency in Hz for the low-pass filter.
    :param fs: Sampling frequency in Hz.
    :param order: Filter order for the Butterworth filter.
    :return: Nx3 NumPy array of filtered accelerations.
    """
    filtered = np.zeros_like(accel_data)
    # Filter each axis (column) independently
    for axis in range(3):
        filtered[:, axis] = butter_lowpass_filter(
            accel_data[:, axis],
            cutoff=cutoff,
            fs=fs,
            order=order
        )
    return filtered

def detect_large_motions(filtered_data, angle_threshold=5e-3):
    """
    Identify samples that contain large (non-breathing) body motions by comparing
    consecutive 3D acceleration vectors.
    :param filtered_data: Nx3 array of filtered accelerometer data.
    :param angle_threshold: Rad/sample threshold above which we classify as a 'large motion'.
    :return: Boolean array of length N, where True indicates 'static' and False indicates 'large motion'.
    """
    n_samples = len(filtered_data)
    # Boolean mask: True = static, False = motion
    # Start by assuming static, then mark large motions as False
    is_static = np.ones(n_samples, dtype=bool)

    for t in range(1, n_samples):
        # Normalize the vectors (t-1 and t)
        prev_vec = filtered_data[t - 1] / np.linalg.norm(filtered_data[t - 1])
        curr_vec = filtered_data[t]     / np.linalg.norm(filtered_data[t])

        # Dot product -> angle
        dot_val = np.dot(prev_vec, curr_vec)
        # Numerical clip in case of floating-point slight overflow
        dot_val = np.clip(dot_val, -1.0, 1.0)
        theta = np.arccos(dot_val)  # Angle in radians

        # Mark as 'not static' if the angle exceeds threshold
        if theta > angle_threshold:
            is_static[t] = False

    return is_static