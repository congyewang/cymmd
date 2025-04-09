import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def median_trick(np.ndarray[DTYPE_t, ndim=2] x):
    """
    The median trick, used to automatically determine the sigma parameter of a Gaussian kernel

    Args:
        x (np.ndarray): The input data, with a shape of [n, dim]

    Returns:
        float: Values suitable for the width of the Gaussian kernel
    """
    cdef int n = x.shape[0]
    cdef int d = x.shape[1]
    cdef int i, j, k
    cdef double dist, median_value
    cdef double *distances
    cdef int n_distances = n * (n - 1) // 2  # Number of elements in an upper triangular matrix

    # Allocate memory for storing distances
    distances = <double*>malloc(n_distances * sizeof(double))
    if not distances:
        raise MemoryError("Memory could not be allocated for distance calculation.")

    # Calculate the distances between all pairs of points
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for d in range(x.shape[1]):
                dist += (x[i, d] - x[j, d]) ** 2
            distances[k] = dist ** 0.5
            k += 1

    # Convert the distance to a numpy array to use np.median
    cdef np.ndarray[DTYPE_t, ndim=1] np_distances = np.empty(n_distances, dtype=DTYPE)
    for i in range(n_distances):
        np_distances[i] = distances[i]

    # Free up memory
    free(distances)

    # Return half of the median value
    return 0.5 * np.median(np_distances).item()

def mmd_block_calculation(
    np.ndarray[DTYPE_t, ndim=2] x,
    np.ndarray[DTYPE_t, ndim=2] y,
    double sigma,
    int block_size=1000
):
    """
    Compute MMD using a chunked calculation approach to reduce memory usage

    Args:
        x (np.ndarray): The first sample set, with a shape of [n_x, dim]
        y (np.ndarray): The second sample set, with shape [n_y, dim]
        sigma (float): The bandwidth parameter of the Gaussian kernel
        block_size (int): Block size for each calculation

    Returns:
        float: The value of MMD^2
    """
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef double mmd_xx = 0.0
    cdef double mmd_yy = 0.0
    cdef double mmd_xy = 0.0
    cdef double beta = 1.0 / (2.0 * sigma * sigma)
    cdef int i, j, k, l
    cdef double dist, kernel_val

    # Compute the x-x part in chunks
    for i in range(0, nx, block_size):
        i_end = min(i + block_size, nx)
        for j in range(i, nx, block_size):  # Starting from i, utilizing symmetry
            j_end = min(j + block_size, nx)

            # Calculate the kernel value within this block
            for k in range(i, i_end):
                for l in range(j if i==j else j, j_end):  # If it is the same block, avoid duplicate calculations.
                    if k == l:  # elements on the diagonal
                        mmd_xx += 1.0  # exp(0) = 1
                        continue

                    dist = 0.0
                    for d in range(x.shape[1]):
                        dist += (x[k, d] - x[l, d]) ** 2

                    kernel_val = exp(-beta * dist)
                    mmd_xx += kernel_val * (2.0 if k != l else 1.0)  # If not on the diagonal, due to symmetry, it needs to be calculated twice.

    # Compute the y-y component in chunks
    for i in range(0, ny, block_size):
        i_end = min(i + block_size, ny)
        for j in range(i, ny, block_size):
            j_end = min(j + block_size, ny)

            # Calculate the kernel value within this block
            for k in range(i, i_end):
                for l in range(j if i==j else j, j_end):
                    if k == l:
                        mmd_yy += 1.0
                        continue

                    dist = 0.0
                    for d in range(y.shape[1]):
                        dist += (y[k, d] - y[l, d]) ** 2

                    kernel_val = exp(-beta * dist)
                    mmd_yy += kernel_val * (2.0 if k != l else 1.0)

    # Compute the x-y part in chunks
    for i in range(0, nx, block_size):
        i_end = min(i + block_size, nx)
        for j in range(0, ny, block_size):
            j_end = min(j + block_size, ny)

            # Calculate the kernel value within this block
            for k in range(i, i_end):
                for l in range(j, j_end):
                    dist = 0.0
                    for d in range(x.shape[1]):
                        dist += (x[k, d] - y[l, d]) ** 2

                    kernel_val = exp(-beta * dist)
                    mmd_xy += kernel_val

    # Calculate the final MMD value
    return (mmd_xx / (nx * nx) + mmd_yy / (ny * ny) - 2.0 * mmd_xy / (nx * ny))

def calculate_mmd(
    np.ndarray[DTYPE_t, ndim=2] x,
    np.ndarray[DTYPE_t, ndim=2] y,
    sigma=None,
    int block_size=1000,
    bint use_median_trick=True
):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions

    Args:
        x (np.ndarray): The first sample set, with a shape of [n_x, dim]
        y (np.ndarray): The second sample set, with shape [n_y, dim]
        sigma (float): The bandwidth parameter of the Gaussian kernel. If None and use_median_trick=True, it is automatically determined using the median trick.
        block_size (int): The block size for each computation, used to reduce memory usage
        use_median_trick: Whether to use the median trick to automatically determine sigma

    Returns:
        float: The value of MMD^2
    """
    cdef double sigma_value

    if sigma is None and use_median_trick:
        # Merge data for the median technique
        combined = np.vstack([x, y])
        sigma_value = median_trick(combined)
    elif sigma is None:
        sigma_value = 1.0
    else:
        sigma_value = float(sigma)  # Ensure that sigma is of type double.

    return mmd_block_calculation(x, y, sigma_value, block_size)
