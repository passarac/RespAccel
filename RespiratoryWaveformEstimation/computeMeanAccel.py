import numpy as np

def compute_mean_accel(accel_data, window_size=30):
    """
    Compute a windowed average of the 3D accelerometer data for each sample index t.
    This serves as the estimated gravity vector (āₜ).

    :param accel_data: Nx3 numpy array of accelerations (ideally normalized).
    :param window_size: Number of samples in the sliding window.
    :return: Nx3 numpy array of window-averaged accelerations (āₜ for each t).
    """
    n_samples = accel_data.shape[0]
    half_w = window_size // 2

    mean_accel = np.zeros_like(accel_data)

    for t in range(n_samples):
        start = max(0, t - half_w)
        end   = min(n_samples, t + half_w + 1)

        # Simple rectangular window average:
        window_slice = accel_data[start:end]
        avg_vec = np.mean(window_slice, axis=0)

        # Normalize the average so it's a unit vector (optional but recommended)
        norm_val = np.linalg.norm(avg_vec)
        if norm_val > 1e-12:
            avg_vec = avg_vec / norm_val

        mean_accel[t] = avg_vec

    return mean_accel