import numpy as np

from .computeRotationAxes import *
from .findReferenceAxis import *

def track_rotation_axis(accel_data, window_size=32, angle_weight=True):
    """
    For each time t, compute a smoothed/averaged rotation axis
    over a neighborhood around t, possibly weighting each sample
    by the angle between a_{t-1} and a_{t}.

    Steps:
      1) Compute instantaneous rotation axes r_t = a_t x a_{t-1}
      2) Find a reference axis (via PCA) to ensure consistent sign
      3) Smooth the axes over a sliding window, weighting by:
           - A Hamming window
           - The angle theta_t (optional)
      4) Normalize each result to get a unit axis

    :param accel_data: Nx3 array of normalized acceleration vectors (static only).
    :param window_size: Number of samples for the sliding window (e.g. 32).
    :param angle_weight: If True, weight by the angle between consecutive vectors.
    :return: Nx3 array of smoothed, unit rotation axes for each sample.
    """
    # 1) Compute all instantaneous rotation axes
    rotation_axes = compute_rotation_axes(accel_data)
    n_samples = rotation_axes.shape[0]

    # 2) Determine reference axis via PCA
    r_ref = find_reference_axis(rotation_axes)

    # 3) We also need angles for weighting, if desired
    angles = np.zeros(n_samples)
    for t in range(1, n_samples):
        # Dot product -> angle
        prev_vec = accel_data[t - 1] / np.linalg.norm(accel_data[t - 1])
        curr_vec = accel_data[t]     / np.linalg.norm(accel_data[t])
        dot_val = np.dot(prev_vec, curr_vec)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angles[t] = np.arccos(dot_val)

    # Prepare a Hamming window for weighting
    ham_window = np.hamming(window_size)  # shape (window_size,)
    half_w = window_size // 2

    # Output array for the smoothed axes
    smoothed_axes = np.zeros_like(rotation_axes)

    for t in range(n_samples):
        # Sliding window boundaries
        start = max(0, t - half_w)
        end   = min(n_samples, t + half_w + 1)

        # Weighted sum
        weighted_sum = np.zeros(3, dtype=float)
        w_total = 0.0

        # Index within the Hamming window
        for i in range(start, end):
            w_idx = i - (t - half_w)  # index into the ham_window
            if w_idx < 0 or w_idx >= window_size:
                continue

            # Basic Hamming weight
            w_hamming = ham_window[w_idx]

            # Optionally multiply by angle
            w_angle = angles[i] if angle_weight else 1.0

            # Combine
            w = w_hamming * w_angle

            # Get the raw axis, flip sign if needed to match r_ref hemisphere
            r_i = rotation_axes[i]
            if np.dot(r_i, r_ref) < 0:
                r_i = -r_i

            weighted_sum += w * r_i
            w_total += w

        # Normalize
        if w_total > 1e-12:  # to avoid divide-by-zero
            smoothed_axis = weighted_sum / w_total
            norm_val = np.linalg.norm(smoothed_axis)
            if norm_val > 1e-12:
                smoothed_axis /= norm_val
            smoothed_axes[t] = smoothed_axis

    return smoothed_axes