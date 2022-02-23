"""
Util module.

General methods to use, no dependencies to other internal modules

"""
import numpy as np
import os
import cv2
import sys
import logging
import multiprocessing as mp
# Alternative implementation in Cython
from brescount import bres_line


def logger():
    """Get the module logger"""
    return logging.getLogger("copa_map")


def get_cpu_count():
    """Get the number of CPUs"""
    try:
        cpu_count = int(os.environ['PBS_NUM_PPN'])  # luis cluster PBS
    except KeyError:
        try:
            cpu_count = int(os.environ['SLURM_CPUS_PER_TASK'])  # luis cluster SLURM
        except KeyError:
            cpu_count = mp.cpu_count()
    # Return one less core to keep the system responsive
    return max([cpu_count - 1, 1])


def _bresenhamline_nslope(slope):
    """Normalize slope for Bresenham's line algorithm."""
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """Returns npts lines of length max_iter each. (npts x max_iter x dimension)."""
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

# def _bresenhamline(start, end):
#     return bres_line(start, end)


def min_on_line_coord(start : np.array, end : np.array, matrix : np.array, thresh : float):
    """
    Return the coordinate on a line between two points, that has a value smaller than a threshold

    Given a start and end point, as coordinates of a matrix, the functions creates a line (bresenham) between the
    two points.
    Then the values of the matrix corresponding to the line are checked, and the first coord. is returned that has
    a value smaller than thresh
    Args:
        start: start point (2 x 1)
        end:  end point (2 x 1)
        matrix: Array with values
        thresh: Threshold value to check

    Returns:
        (2 x 1) coordinate with first small value, or endpoint if none is found
    """
    assert len(start) == 2 and len(end) == 2, "Coordinates must be 2D arrays"
    line = bres_line(start, end)
    vals_line = matrix[line[:, 0], line[:, 1]]
    cond = vals_line < thresh
    # If no value is smaller, return endpoint
    if np.all(~cond):
        return end
    # Return first point where value is smaller
    return line[np.argmax(cond)]


def bresenhamline(start, end, max_iter=-1):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the points.

    Args:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.
    """
    # Return the points as a single array
    # return _bresenhamline(start, end)
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def isPD(matrix):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    Taken from: https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A = np.array(A)
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I_diag = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I_diag * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def package_path(*paths, file=__file__):
    """
    Helper to receive an absolute path within the package directories

    Args:
        *paths: (Optional) String to append to package path
        file: (Optional) File to get the path from, default parameter this script

    Returns:
        Absolute path
    """
    return os.path.join(os.path.dirname(os.path.abspath(file)), *paths)


def abs_path():
    """Absolute package path"""
    return os.path.abspath(os.path.join(sys.modules["copa_map"].__file__, os.pardir))


def remove_small_objects_in_image(img, min_pixel_size):
    """
    Removes white objects, which are smaller than minimum size

    Args:
        img: Input image
        min_pixel_size: Minimum pixel size

    Returns:
        Cleaned image
    """
    img = img.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_pixel_size:
            img2[output == i + 1] = 255
    return img2.astype(np.uint8)
