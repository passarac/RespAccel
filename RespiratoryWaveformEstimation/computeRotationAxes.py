import numpy as np

def compute_rotation_axes(accel_data):
    """
    Compute the instantaneous rotation axes r_t = a_t x a_{t-1}
    for each pair of consecutive accelerations.

    :param accel_data: Nx3 numpy array of normalized acceleration vectors,
                       e.g. after low-pass filtering & selecting static segments.
    :return: (N x 3) array of rotation axes, r[0] = [0,0,0] for convenience,
             because there's no previous sample at t=0.
    """
    n_samples = accel_data.shape[0]
    rotation_axes = np.zeros_like(accel_data)

    for t in range(1, n_samples):
        # Cross product of consecutive vectors
        rotation_axes[t] = np.cross(accel_data[t], accel_data[t - 1])

    return rotation_axes