import numpy as np

def compute_breathing_angle(accel_data, mean_accel, smoothed_axes):
    """
    Compute the breathing angle φₜ = arcsin( ((āₜ × r̄ₜ) ⋅ aₜ) )
    for each time t.

    :param accel_data: Nx3 array of accelerations (aₜ).
    :param mean_accel: Nx3 array of mean accelerations (āₜ), e.g. from compute_mean_accel().
    :param smoothed_axes: Nx3 array of smoothed rotation axes (r̄ₜ), from Step 2.
    :return: 1D array of length N giving φₜ in radians.
    """
    n_samples = accel_data.shape[0]
    phi = np.zeros(n_samples, dtype=float)

    for t in range(n_samples):
        # Cross product: (āₜ × r̄ₜ)
        cross_vec = np.cross(mean_accel[t], smoothed_axes[t])

        # Dot with aₜ
        dot_val = np.dot(cross_vec, accel_data[t])

        # arcsin( ... ), with clipping to [-1, 1] in case of floating-point tiny overshoot
        dot_val = np.clip(dot_val, -1.0, 1.0)
        phi[t] = np.arcsin(dot_val)

    return phi