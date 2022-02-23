import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdlib cimport abs as cabs
cimport openmp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long bres_segment_count(long x0, long y0,
                            long x1, long y1, long* xind, long* yind) nogil:
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    cdef unsigned inc_ind
    cdef long e2, sx, sy, err
    cdef long dx, dy, nrows, ncols
    inc_ind = 0
    nrows = cabs(x1-x0) + 1
    ncols = cabs(y1-y0) + 1
    # print (" nrows " + str(nrows) + " ncols " + str(ncols))
    if x1 > x0:
        dx = x1 - x0
    else:
        dx = x0 - x1
    if y1 > y0:
        dy = y1 - y0
    else:
        dy = y0 - y1

    sx = 0
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    sy = 0
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        xind[inc_ind] = x0
        yind[inc_ind] = y0

        if x0 == x1 and y0 == y1:
            break

        inc_ind = inc_ind + 1

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    inc_ind = inc_ind + 1
    return inc_ind

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef bres_line(long[:] p0, long[:] p1, long MAX_LEN = 10000):
    """
    Create a Bresenham line between two 2d int point pairs
    Args:
        p0: First point
        p1:  Second point
        MAX_LEN: Maximal number of elements between the two index pairs (used for allocation)

    Returns:
        Array of [n x 2] dimension containing all cells between the two points
    """
    cdef unsigned k
    cdef long x0, y0, x1, y1, x_grid, y_grid, xb, yb
    cdef bint flip
    # It should be ensured that the same index pairs results, if the start and end of the line are switched
    # Bresenham's algorithm does not ensure this though. Therefore, we switch start and end if x1 > x0
    # At the end, we switch the order again
    if p0[0]> p1[0]:
        flip = True
        x0 = p1[0]
        y0 = p1[1]
        x1 = p0[0]
        y1 = p0[1]
    else:
        flip = False
        x0 = p0[0]
        y0 = p0[1]
        x1 = p1[0]
        y1 = p1[1]

    xind = np.ascontiguousarray(np.ones(MAX_LEN, dtype=np.int64) * -1)
    yind = np.ascontiguousarray(np.ones(MAX_LEN, dtype=np.int64) * -1)
    cdef long[::1] xind_view, yind_view
    xind_view = xind
    yind_view = yind

    num_ind = bres_segment_count(x0, y0, x1, y1, &xind_view[0], &yind_view[0])
    ind = np.vstack([xind, yind])
    if flip:
        return np.flipud(ind[:,:num_ind].T)
    else:
        return ind[:, :num_ind].T

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef double _iter_matrix(long[:, :] p_ind, double[:, ::1] mat, double min_val,
                  long MAX_LEN, Py_ssize_t idx_row, long* xind, long* yind, bint use_multiproc) nogil:
    """
    Function to check a specific index pair.
    
    Paremeter 'idx_row' determines which row of 'p_ind' to check
    Args:
        p_ind: [n x 4] Array containing the pairs to check. Every row has the form [x0, y0, x1, y1]
        mat: [i x j] Matrix to use the values from
        min_val: Minimal value. If a value of the matrix is below this value, this value will be assumed
        MAX_LEN: Maximal number of elements between the two index pairs (used for allocation)
        idx_row: Row index of 'p_ind' to check
        xind: Buffer array for x values of line segments
        yind: Buffer array for y values of line segments

    Returns:
        Minimum value on line between row indices
    """
    cdef:
        long x0, y0, x1, y1
        int tid
        double temp_min, val
        long lx, ly
        unsigned inc_ind
        Py_ssize_t idx_line
    # Thread number to access correct section of buffer arrays
    if use_multiproc:
        tid = openmp.omp_get_thread_num()
    else:
        tid = 0
    # It should be ensured that the same index pairs results, if the start and end of the line are switched
    # Bresenham's algorithm does not ensure this though. Therefore, we switch start and end if x1 > x0
    if p_ind[idx_row, 0] > p_ind[idx_row, 2]:
        x0 = p_ind[idx_row, 2]
        y0 = p_ind[idx_row, 3]
        x1 = p_ind[idx_row, 0]
        y1 = p_ind[idx_row, 1]
    else:
        x0 = p_ind[idx_row, 0]
        y0 = p_ind[idx_row, 1]
        x1 = p_ind[idx_row, 2]
        y1 = p_ind[idx_row, 3]
    # Count the number of cells between the index pairs. Pairs are written to xind and yind
    inc_ind = bres_segment_count(x0, y0, x1, y1, &xind[tid * MAX_LEN], &yind[tid * MAX_LEN])
    temp_min = 10000001
    # Iterate on the line (index pairs)
    for idx_line in range(inc_ind):
        lx = xind[idx_line + tid * MAX_LEN]
        ly = yind[idx_line + tid * MAX_LEN]
        val = mat[lx, ly]
        # We search for the minimum
        if val < temp_min:
            temp_min = val
            if temp_min <= min_val:
                temp_min = min_val
       # print("lx " + str(lx) + " ly " + str(ly) + " val " + str(val) + " temp_min " + str(temp_min))

    return temp_min

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cpdef matrix_minima_on_line(long[:, :] p_ind, double[:, ::1] mat, double min_val, long MAX_LEN = 10000,
                            int num_threads = 1):
    """
    Function to calculate the minimal value on a straight line between two matrix index pairs
    
    Based on Bresenham's algorithm, determines which entries of the matrix ly between the two index pairs and checks
    which is the minimal value on the direct line
    Args:
        p_ind: [n x 4] Array containing the pairs to check. Every row has the form [x0, y0, x1, y1]
        mat: [i x j] Matrix to use the values from
        min_val: Minimal value. If a value of the matrix is below this value, this value will be assumed
        MAX_LEN: Maximal number of elements between the two index pairs (used for allocation)
        num_threads: Number of threads for multiprocessing. Multiprocessing usually leads to a strong performance increase
                     If =1, normal range loop will be used instead of Cython's prange
    Returns:
        [n x 1] Vector with minimal matrix values
    """

    cdef:
        Py_ssize_t arr_len = p_ind.shape[0]
        Py_ssize_t idx_row

    # For very small arrays single processing is a bit faster (less overhead)
    # Therefore, single processing is used if array is smaller than threshold (determined empirically)
    if arr_len < 20000:
        num_threads = 1

    cdef:
        # Allocate arrays where the indices of line elements will be written to
        # Each process uses a segment of this array (leads to significant performance increase)
        long[::1] xind = np.ascontiguousarray(np.ones(MAX_LEN * num_threads, dtype=np.int64) * -1)
        long[::1] yind = np.ascontiguousarray(np.ones(MAX_LEN * num_threads, dtype=np.int64) * -1)

    # Create array which will be returned
    arr_min = np.empty(arr_len, dtype=np.double)
    # Memory view to efficiently access the array data
    cdef double[::1] arr_min_view = arr_min

    # # First case: Multiprocessing.
    if num_threads > 1:
        for idx_row in prange(arr_len, nogil=True, schedule='static', num_threads=num_threads):
            arr_min_view[idx_row] =  _iter_matrix(p_ind, mat, min_val, MAX_LEN, idx_row, &xind[0], &yind[0], True)
    # Second case: Use single processing.
    else:
        for idx_row in range(arr_len):
            arr_min_view[idx_row] =  _iter_matrix(p_ind, mat, min_val, MAX_LEN, idx_row, &xind[0], &yind[0], False)

    return arr_min


