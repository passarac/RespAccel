import numpy as np

def find_reference_axis(rotation_axes):
    """
    Estimate a single 'reference axis' by taking the
    first principal component (largest variance direction)
    of all the rotation axes.

    :param rotation_axes: Nx3 array of raw rotation axes.
    :return: A unit vector in R^3 that represents the primary axis direction.
    """
    # Remove any zero rows (which can happen at t=0 or if cross product was zero).
    valid = np.any(rotation_axes != 0, axis=1)
    valid_axes = rotation_axes[valid]

    if len(valid_axes) < 2:
        # Fallback: just return a default axis if there's not enough data
        return np.array([0, 0, 1], dtype=float)

    # Subtract mean
    mean_axis = np.mean(valid_axes, axis=0)
    centered = valid_axes - mean_axis

    # Perform SVD on the centered axes
    # The first right-singular vector (V[0]) is the principal component
    u, s, vh = np.linalg.svd(centered, full_matrices=False)

    # The first principal component is vh[0], but we typically want a unit vector
    pc1 = vh[0]
    pc1_norm = pc1 / np.linalg.norm(pc1)

    return pc1_norm